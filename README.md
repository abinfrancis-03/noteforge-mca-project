# Noteforge — Admin Module Setup

## 1. Install dependencies
```bash
pip install -r requirements.txt
```

## 2. Run migrations
```bash
python manage.py makemigrations accounts
python manage.py makemigrations
python manage.py migrate
```

## 3. Create superuser (Admin account)
```bash
python manage.py shell
```
Then paste:
```python
from accounts.models import User
User.objects.create_superuser(
    username='admin',
    password='admin123',
    email='admin@noteforge.com',
    role='admin',
    is_first_login=False
)
exit()
```

## 4. Run server
```bash
python manage.py runserver
```

## 5. Open browser
- Login: http://127.0.0.1:8000/login/
- Username: `admin`  |  Password: `admin123`

---

## Admin Module covers:
- Dashboard with stats
- Departments — Add / Edit / Delete
- Semesters — Add / Edit / Delete (filtered by dept)
- Subjects — Add / Edit / Delete (filtered by semester)
- Teachers — Add / Edit / Delete / Reset Password
- Students — Add / Edit / Delete / Reset Password
- Audit Log — full action trail with filter
