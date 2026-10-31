import os
import re
import math
import time
import requests
from typing import List, Dict, Any, Tuple, Optional, Iterable

from django.conf import settings
from django.utils import timezone
from accounts.models import StudyMaterial, StudyMaterialChunk, Subject, User
from services.document_extractor.extractor_factory import get_extractor
from services.document_extractor.base import (
    NormalizedDocument,
    NormalizedSection,
    UnsupportedFileFormatException,
    EmptyDocumentException,
    DocumentExtractionException,
)


def get_gemini_api_key() -> str:
    """Load Gemini API Key from environment or .env file."""
    env_file = getattr(settings, 'BASE_DIR', None)
    if env_file:
        env_path = env_file / '.env'
        if env_path.exists():
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            os.environ[k.strip()] = v.strip().strip('"').strip("'")
            except Exception:
                pass
    return (
        os.environ.get('GEMINI_API_KEY')
        or getattr(settings, 'GEMINI_API_KEY', '')
        or os.environ.get('GOOGLE_API_KEY')
        or ''
    )


# Common stop words for keyword extraction
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'aren\'t',
    'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
    'can', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing',
    'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t',
    'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if',
    'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 'let\'s', 'me', 'more', 'most',
    'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other',
    'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll',
    'she\'s', 'should', 'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their',
    'theirs', 'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll',
    'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 'what', 'what\'s', 'when',
    'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with',
    'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours',
    'yourself', 'yourselves', 'tell', 'explain', 'give', 'define', 'describe', 'briefly', 'detail'
}


class RAGEngine:
    """
    Production-grade Retrieval-Augmented Generation (RAG) Engine for NoteForge.
    Grounded strictly in teacher-uploaded study materials available to the authenticated student.
    Enforces 'NO SOURCE -> NO ANSWER' anti-hallucination guardrails.
    """

    EMBEDDING_MODEL = 'gemini-embedding-001'
    GENERATION_MODELS = [
        'gemini-flash-lite-latest',
        'gemini-3.5-flash-lite',
        'gemini-3.1-flash-lite',
        'gemini-flash-latest',
        'gemini-3.8-flash',
        'gemini-3.7-flash',
        'gemini-3.5-flash',
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_gemini_api_key()

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Embeddings & Mathematical Similarity
    # ──────────────────────────────────────────────────────────────────────────

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embedding vectors using Gemini batchEmbedContents API.
        Falls back to deterministic TF-IDF term hashing if API is unavailable.
        """
        if not texts:
            return []

        if not self.api_key:
            return [self._fallback_sparse_vector(t) for t in texts]

        results: List[List[float]] = []
        batch_size = 25

        for i in range(0, len(texts), batch_size):
            chunk_batch = texts[i:i + batch_size]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.EMBEDDING_MODEL}:batchEmbedContents?key={self.api_key}"
            requests_payload = [
                {
                    "model": f"models/{self.EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": t[:2048]}]}
                }
                for t in chunk_batch
            ]
            payload = {"requests": requests_payload}

            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    for emb in embeddings:
                        results.append(emb.get("values", []))
                else:
                    # Fallback for this batch
                    for t in chunk_batch:
                        results.append(self._fallback_sparse_vector(t))
            except Exception:
                for t in chunk_batch:
                    results.append(self._fallback_sparse_vector(t))

        return results

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generates single embedding vector for the student query.
        """
        if not self.api_key or not query.strip():
            return self._fallback_sparse_vector(query)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.EMBEDDING_MODEL}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.EMBEDDING_MODEL}",
            "content": {"parts": [{"text": query[:2048]}]}
        }

        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                return resp.json().get("embedding", {}).get("values", [])
        except Exception:
            pass

        return self._fallback_sparse_vector(query)

    @staticmethod
    def _fallback_sparse_vector(text: str, dim: int = 128) -> List[float]:
        """
        Deterministic hashed term frequency vector fallback.
        Ensures search never crashes even during network outages.
        """
        vec = [0.0] * dim
        words = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())
        for w in words:
            if w not in STOP_WORDS:
                idx = abs(hash(w)) % dim
                vec[idx] += 1.0
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Computes cosine similarity between two float vectors."""
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            # If dimensions mismatch (e.g. dense vs fallback sparse), normalize lengths
            min_len = min(len(vec_a), len(vec_b))
            vec_a = vec_a[:min_len]
            vec_b = vec_b[:min_len]

        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for a, b in zip(vec_a, vec_b):
            dot += a * b
            norm_a += a * a
            norm_b += b * b

        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract meaningful alphanumeric words from text excluding common stop words."""
        tokens = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())
        return [t for t in tokens if t not in STOP_WORDS]

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Material Ingestion, Chunking & Synchronization
    # ──────────────────────────────────────────────────────────────────────────

    def index_study_material(self, material: StudyMaterial, force_reindex: bool = False) -> int:
        """
        Extracts, chunks, embeds, and indexes a single StudyMaterial into StudyMaterialChunk.
        Returns the number of indexed chunks.
        """
        if not material.file or not os.path.exists(material.file.path):
            return 0

        # Check existing
        existing_count = material.chunks.count()
        if existing_count > 0 and not force_reindex:
            return existing_count

        if force_reindex and existing_count > 0:
            material.chunks.all().delete()

        try:
            extractor = get_extractor(material.file)
            doc: NormalizedDocument = extractor.extract(material.file, document_title=material.title)
        except Exception as e:
            return 0

        if not doc.sections:
            return 0

        # Prepare semantic chunks
        chunk_data: List[Dict[str, Any]] = []
        chunk_idx = 1

        for sec in doc.sections:
            title = (sec.title or '').strip()
            content = (sec.content or '').strip()
            if not content and not title:
                continue

            ref = sec.source_reference or {}
            unit_type = ref.get('type', 'page')
            unit_number = ref.get('number', chunk_idx)

            combined_text = f"[{unit_type.upper()} {unit_number}] {title}\n{content}".strip()
            if len(combined_text) < 25:
                # Skip trivial headers/numbers
                continue

            # Split excessively long sections into sub-chunks of ~1200 chars
            if len(content) > 1800:
                paragraphs = re.split(r'\n{2,}|\.\s+', content)
                cur_text = f"[{unit_type.upper()} {unit_number}] {title}\n"
                sub_idx = 1
                for para in paragraphs:
                    if len(cur_text) + len(para) > 1200:
                        chunk_data.append({
                            'chunk_index': chunk_idx,
                            'title': f"{title} (Part {sub_idx})" if sub_idx > 1 else title,
                            'unit_type': unit_type,
                            'unit_number': unit_number,
                            'content': cur_text.strip(),
                        })
                        chunk_idx += 1
                        sub_idx += 1
                        cur_text = f"[{unit_type.upper()} {unit_number}] {title} (Cont.)\n{para}\n"
                    else:
                        cur_text += f"{para}\n"
                if len(cur_text.strip()) > 35:
                    chunk_data.append({
                        'chunk_index': chunk_idx,
                        'title': f"{title} (Part {sub_idx})" if sub_idx > 1 else title,
                        'unit_type': unit_type,
                        'unit_number': unit_number,
                        'content': cur_text.strip(),
                    })
                    chunk_idx += 1
            else:
                chunk_data.append({
                    'chunk_index': chunk_idx,
                    'title': title or f"{unit_type.capitalize()} {unit_number}",
                    'unit_type': unit_type,
                    'unit_number': unit_number,
                    'content': combined_text,
                })
                chunk_idx += 1

        if not chunk_data:
            return 0

        # Generate embeddings in batch
        texts_to_embed = [c['content'] for c in chunk_data]
        embeddings = self.generate_embeddings_batch(texts_to_embed)

        # Bulk create chunks
        instances = []
        for i, c in enumerate(chunk_data):
            emb = embeddings[i] if i < len(embeddings) else []
            instances.append(
                StudyMaterialChunk(
                    material=material,
                    chunk_index=c['chunk_index'],
                    title=c['title'][:250],
                    unit_type=c['unit_type'][:45],
                    unit_number=c['unit_number'],
                    content=c['content'],
                    char_count=len(c['content']),
                    token_count=len(c['content']) // 4,
                    embedding=emb,
                )
            )

        StudyMaterialChunk.objects.bulk_create(instances)
        return len(instances)

    def ensure_materials_indexed(self, materials: Iterable[StudyMaterial]) -> None:
        """Verifies all specified materials have chunks indexed in DB; indexes any missing ones."""
        for mat in materials:
            if mat.chunks.count() == 0 and mat.file:
                self.index_study_material(mat)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Student Authorization Scoping & Semantic Retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def get_authorized_materials(self, student: User, subject_id: Optional[Any] = None) -> List[StudyMaterial]:
        """
        Enforces strict student authorization boundaries:
        - Accessible to the authenticated student's department up to their current semester (or enrolled semester).
        - Optionally filtered by a specific subject.
        """
        if not student or not student.is_authenticated or student.role != 'student':
            return []

        qs = StudyMaterial.objects.all()
        if student.department:
            qs = qs.filter(subject__semester__department=student.department)

        if student.semester and hasattr(student.semester, 'number') and student.semester.number:
            qs = qs.filter(subject__semester__number__lte=student.semester.number)
        elif student.semester:
            qs = qs.filter(subject__semester=student.semester)

        if subject_id and str(subject_id).lower() not in ('all', '', 'none', '0'):
            try:
                qs = qs.filter(subject_id=int(subject_id))
            except (ValueError, TypeError):
                pass

        return list(qs.select_related('subject', 'uploaded_by'))

    def retrieve_relevant_chunks(
        self,
        student: User,
        query: str,
        subject_id: Optional[Any] = None,
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Retrieves top relevant chunks strictly from authorized materials using Hybrid Search:
        Combines Dense Cosine Similarity + Lexical Keyword Overlap.
        Enforces strict relevance gating against out-of-scope queries.
        """
        authorized_materials = self.get_authorized_materials(student, subject_id)
        if not authorized_materials:
            return [], {"status": "no_materials", "reason": "No study materials available for your semester/department."}

        # Ensure chunks are indexed
        self.ensure_materials_indexed(authorized_materials)

        chunks_qs = StudyMaterialChunk.objects.filter(
            material__in=authorized_materials
        ).select_related('material', 'material__subject')

        if not chunks_qs.exists():
            return [], {"status": "no_chunks", "reason": "No text content available in uploaded materials."}

        # 1. Compute query vector & query keywords
        q_clean = query.strip()
        q_emb = self.generate_query_embedding(q_clean)
        q_keywords = self.extract_keywords(q_clean)
        q_phrase = q_clean.lower()

        # 2. Score candidate chunks
        scored_candidates: List[Dict[str, Any]] = []

        for chunk in chunks_qs:
            # Cosine similarity
            cos_sim = self.cosine_similarity(q_emb, chunk.embedding) if chunk.embedding else 0.0

            # Lexical keyword score
            c_lower = chunk.content.lower()
            t_lower = chunk.title.lower()

            keyword_hits = 0
            for kw in q_keywords:
                if kw in c_lower or kw in t_lower:
                    keyword_hits += 1

            keyword_ratio = (keyword_hits / len(q_keywords)) if q_keywords else 0.0

            # Exact phrase bonus (e.g. "polymorphism", "data representation")
            phrase_bonus = 0.25 if (len(q_clean) > 3 and q_phrase in c_lower) else 0.0

            # Hybrid score
            hybrid_score = (0.65 * cos_sim) + (0.35 * keyword_ratio) + phrase_bonus

            scored_candidates.append({
                'chunk': chunk,
                'cos_sim': cos_sim,
                'keyword_hits': keyword_hits,
                'keyword_ratio': keyword_ratio,
                'hybrid_score': hybrid_score,
                'material_title': chunk.material.title,
                'subject_name': chunk.material.subject.name,
                'subject_code': chunk.material.subject.subject_code,
                'unit_type': chunk.unit_type,
                'unit_number': chunk.unit_number,
                'content': chunk.content,
            })

        # Sort descending by hybrid_score
        scored_candidates.sort(key=lambda x: x['hybrid_score'], reverse=True)

        if not scored_candidates:
            return [], {"status": "not_found", "reason": "No matching content found."}

        best_match = scored_candidates[0]
        # Anti-Hallucination Relevance Gate:
        # A query is considered covered in the material if:
        # 1. At least one significant keyword appears in the matched notes and cos_sim >= 0.46
        # 2. OR semantic cosine similarity is exceptionally high (>= 0.68)
        has_keyword_match = best_match['keyword_hits'] > 0
        has_strong_semantic = best_match['cos_sim'] >= 0.68
        has_moderate_match = best_match['cos_sim'] >= 0.46 and has_keyword_match

        if not (has_strong_semantic or has_moderate_match):
            return [], {
                "status": "not_found",
                "top_score": best_match['hybrid_score'],
                "reason": "Topic not present in student materials."
            }

        # Filter top_k relevant chunks that meet minimum threshold
        selected = [c for c in scored_candidates[:top_k] if c['hybrid_score'] >= 0.32 or c['keyword_ratio'] >= 0.25]
        if not selected:
            selected = [best_match]

        return selected, {"status": "ok", "top_score": best_match['hybrid_score']}

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Anti-Hallucination Answer Generation
    # ──────────────────────────────────────────────────────────────────────────

    def answer_question(
        self,
        student: User,
        query: str,
        subject_id: Optional[Any] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Full RAG pipeline:
        1. Validates query.
        2. Retrieves relevant chunks scoped strictly to student materials.
        3. Gates out of scope queries with polite declination.
        4. Synthesizes structured grounded answer with 📚 Sources citations.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "is_grounded": True,
                "status": "empty_query"
            }

        # Extract topic from query for clear declination messages
        topic_match = re.sub(r'^(what is|what are|explain|describe|tell me about|how does|why is|define)\s+', '', cleaned_query, flags=re.IGNORECASE)
        topic_match = topic_match.rstrip('?.!').strip()
        display_topic = topic_match if topic_match else cleaned_query

        # Retrieve relevant chunks
        chunks, meta = self.retrieve_relevant_chunks(student, cleaned_query, subject_id=subject_id, top_k=5)

        status = meta.get("status")
        if status == "no_materials":
            return {
                "answer": "You don't have any study materials available yet.",
                "sources": [],
                "is_grounded": True,
                "status": "no_materials"
            }

        if status == "no_chunks" or status == "not_found" or not chunks:
            return {
                "answer": f"Sorry, I couldn't find information about **{display_topic}** in your available study materials.\n\nPlease ask a question related to the materials uploaded by your teacher.",
                "sources": [],
                "is_grounded": True,
                "status": "not_found"
            }

        # Build Context from Retrieved Chunks
        context_blocks = []
        sources_set = set()
        sources_list = []
        represented_material_ids = set()

        for idx, c in enumerate(chunks, 1):
            mat_title = c['material_title']
            utype = c['unit_type'].capitalize()
            unum = c['unit_number']
            sub_code = c['subject_code']
            citation = f"{mat_title} ({utype} {unum})"
            if citation not in sources_set:
                sources_set.add(citation)
                sources_list.append({
                    "material_title": mat_title,
                    "unit_type": utype,
                    "unit_number": unum,
                    "subject_code": sub_code,
                    "citation": citation
                })

            chunk_obj = c.get('chunk')
            if chunk_obj and hasattr(chunk_obj, 'material'):
                mat = chunk_obj.material
                if mat.id not in represented_material_ids and getattr(mat, 'ai_summary', None) and len(mat.ai_summary.strip()) > 30:
                    represented_material_ids.add(mat.id)
                    summary_excerpt = mat.ai_summary.strip()
                    if len(summary_excerpt) > 1000:
                        summary_excerpt = summary_excerpt[:1000] + "..."
                    context_blocks.append(
                        f"[Curriculum Summary & Key Concepts: {mat.title} | Subject: {sub_code}]\n{summary_excerpt}"
                    )

            context_blocks.append(
                f"[Source: {mat_title} | {utype} {unum} | Subject: {sub_code}]\n{c['content']}"
            )

        combined_context = "\n\n---\n\n".join(context_blocks)

        # Format conversation history if available
        history_text = ""
        if conversation_history:
            turns = []
            for h in conversation_history[-3:]:  # Keep last 3 turns
                role = h.get('role', 'user')
                text = h.get('text', '').strip()
                if text:
                    prefix = "Student" if role == 'user' else "AI"
                    turns.append(f"{prefix}: {text}")
            if turns:
                history_text = "RECENT CONVERSATION HISTORY:\n" + "\n".join(turns) + "\n\n"

        # Structured AI Educational Assistant Prompt
        prompt = f"""You are a personalized, private AI Study Assistant for a student.
Your mission is to provide accurate, comprehensive, and high-quality study explanations for topics present in the student's teacher-uploaded study materials.

========================================================
CRITICAL OPERATING RULES:
========================================================
1. TOPIC PRESENCE & STRICT BOUNDARY:
   - Check if the student's topic, term, or concept is mentioned, listed, or covered in the STUDY MATERIAL EXCERPTS below.
   - If the topic or concept IS present in the excerpts (even if as a slide heading, syllabus item, or bullet point, such as waterfall model, software engineering, 1's complement, cohesion, coupling, style guides, etc.), you MUST answer and explain it thoroughly.
   - If the student asks about a topic COMPLETELY OUTSIDE their study materials (e.g. Quantum Computing, Photosynthesis, or any topic with zero presence in the materials), do NOT answer from general knowledge. Decline politely with this EXACT message format:
"Sorry, I couldn't find information about **{display_topic}** in your available study materials.

Please ask a question related to the materials uploaded by your teacher."

2. ADVANCED PEDAGOGICAL STRUCTURING:
   - For any topic present in the study materials, synthesize a crystal-clear, accurate, and structured educational explanation:
     • **Definition & Core Concept**: Clear, authoritative definition and what it is.
     • **How It Works & Key Principles**: Underlying mechanism, characteristics, and rationale.
     • **Steps / Working Process**: Sequential steps or workflow with concrete illustrative calculations/examples where applicable.
     • **Types / Classifications / Comparisons**: Contrast related concepts or list variants (e.g. Adaptive/Sashimi waterfall, 1's vs 2's complement, etc.).
     • **Exam-Critical Points**: Key takeaways, advantages, disadvantages, and exam tips.
   - Do NOT just copy raw, choppy fragments. Synthesize a professional, clear, complete explanation that teaches the student thoroughly!

3. SOURCE CITATIONS:
   - Conclude your response with the exact source materials cited from the excerpts:
📚 Sources:
• [List the study material name and page/slide from the excerpts]

{history_text}STUDY MATERIAL EXCERPTS:
========================================================
{combined_context}
========================================================

STUDENT QUESTION:
{cleaned_query}
"""

        # Call Gemini LLM
        generated_answer, model_used = self._call_gemini_generation(prompt)

        # Check for empty response or API fallback
        if not generated_answer:
            # Fallback direct generation from chunks
            fallback_answer = self._generate_fallback_answer(chunks, cleaned_query, display_topic, sources_list)
            return {
                "answer": fallback_answer,
                "sources": sources_list,
                "is_grounded": True,
                "status": "ok",
                "model": "Local Grounded Engine"
            }

        # Check if LLM flagged that topic was absent
        ans_lower = generated_answer.lower()
        if (
            "couldn't find information" in ans_lower
            or "not covered in your available" in ans_lower
            or "not covered in the study materials" in ans_lower
            or "not mentioned in the excerpts" in ans_lower
            or "outside the scope of" in ans_lower
            or "topic not found" in ans_lower
        ):
            clean_refusal = (
                f"Sorry, I couldn't find information about **{display_topic}** in your available study materials.\n\n"
                f"Please ask a question related to the materials uploaded by your teacher."
            )
            return {
                "answer": clean_refusal,
                "sources": [],
                "is_grounded": True,
                "status": "not_found"
            }

        return {
            "answer": generated_answer,
            "sources": sources_list,
            "is_grounded": True,
            "status": "ok",
            "model": model_used
        }

    def _call_gemini_generation(self, prompt: str) -> Tuple[Optional[str], str]:
        """Calls Gemini API across fallback models."""
        if not self.api_key:
            return None, "No API Key"

        headers = {'Content-Type': 'application/json'}
        for model in self.GENERATION_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.15,
                    "maxOutputTokens": 2048,
                }
            }
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get('candidates', [])
                    if candidates:
                        parts = candidates[0].get('content', {}).get('parts', [])
                        if parts and 'text' in parts[0]:
                            return parts[0]['text'].strip(), f"Google Gemini ({model})"
            except Exception:
                continue

        return None, "API Failure"

    def _generate_fallback_answer(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        topic: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """Rule-based synthesis from top chunks when LLM API is temporarily unreachable."""
        best_chunk = chunks[0]
        content_lines = [l.strip() for l in best_chunk['content'].splitlines() if l.strip() and not l.startswith('[')]
        points = [l for l in content_lines if len(l) > 20][:4]

        lines = [
            f"Based directly on your teacher's uploaded notes for **{best_chunk['material_title']}**:\n"
        ]
        for pt in points:
            clean_pt = pt.lstrip('-*•0123456789. ')
            lines.append(f"• {clean_pt}")

        lines.append("\n📚 Sources:")
        for s in sources:
            lines.append(f"• {s['citation']}")

        return "\n".join(lines)
