# 📚 Noteforge – AI-Powered Learning Platform

**Transforming Study Materials into Smart Learning Experiences**

Noteforge is an AI-powered educational web platform designed to help college students learn more effectively from large PDF documents and educational videos.

The platform provides a centralized learning environment where students can access academic materials, generate AI-powered summaries, interact with a contextual AI tutor, create visual mind maps, and practice through AI-generated tests.

## 🚀 Key Features

### 👨‍💼 Admin Module
- Manage departments, semesters, and subjects.
- Organize academic structures.
- Manage the educational platform.

### 👨‍🏫 Teacher Module
- Manage students using unique university roll numbers.
- Upload and organize study materials subject-wise.
- Create practice tests and MCQ examinations.
- Monitor student test completion and performance.

### 👨‍🎓 Student Module
- Personalized student dashboard based on academic semester.
- Access and download teacher-uploaded study materials.
- Take practice tests and view results.
- Access AI-powered learning tools.

## 🤖 AI-Powered Features

### 1. Document Summarizer
Convert lengthy PDF documents into short, structured, and easy-to-understand summaries.

### 2. YouTube Video Summarizer
Transform educational YouTube videos into structured study notes that can be generated as PDF documents.

### 3. Contextual AI Chatbot
An AI-powered learning assistant that uses Retrieval-Augmented Generation (RAG) to answer questions based on teacher-uploaded academic materials.

### 4. Visual Mind Map Generator
Convert lengthy chapters and study materials into visual mind maps to make learning and revision easier.

### 5. AI-Based MCQ Generator
Generate meaningful multiple-choice questions from uploaded PDF study materials to support practice and assessment.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Backend | Python, Django |
| Frontend | HTML, CSS, JavaScript, Bootstrap |
| Database | SQLite |
| AI Integration | Generative AI, RAG |
| Document Processing | PDF, DOCX, PPTX extraction |
| Version Control | Git, GitHub |

## 📂 Project Structure

```text
noteforge/
│
├── accounts/
├── admin_panel/
├── student_panel/
├── teacher_panel/
├── services/
│   └── document_extractor/
│
├── templates/
│   ├── admin_panel/
│   ├── student_panel/
│   └── teacher_panel/
│
├── core/
├── media/
├── manage.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abinfrancis-03/noteforge-mca-project.git
```

### 2. Navigate to the Project Directory

```bash
cd noteforge-mca-project
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Environment Variables

Create your local `.env` file using `.env.example` as a reference.

Add the required API keys and configuration values.

**Note:** Never upload your actual `.env` file or API keys to GitHub.

### 7. Apply Database Migrations

```bash
python manage.py migrate
```

### 8. Start the Development Server

```bash
python manage.py runserver
```

Open your browser:

```text
http://127.0.0.1:8000/
```

## 🔐 Security

- Role-based access for administrators, teachers, and students.
- Environment variables for sensitive configuration.
- Academic materials organized according to the platform's access structure.

## 🎯 Project Objective

The primary objective of Noteforge is to make academic learning smarter, easier, and more accessible by integrating Artificial Intelligence into the traditional educational system.

## 👨‍💻 Developer

**Abin Francis**

MCA Student | Python Full Stack Developer

- GitHub: https://github.com/abinfrancis-03

---

⭐ If you find this project interesting, consider giving it a star on GitHub!

**Project:** Noteforge – AI-Powered Educational Learning Platform
