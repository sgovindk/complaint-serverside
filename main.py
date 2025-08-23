"""
Project: Agentic AI-powered Campus Complaint Portal – Backend (FastAPI + SQLite)

How to run
----------
1) Create and activate a virtualenv (optional)
2) Install deps:  pip install -r requirements.txt
3) Create a .env file (at project root) with keys:

GEMINI_API_KEY=your_gemini_key_here
ANTI_RAGGING_EMAIL=antiraggingcell@example.edu
FROM_EMAIL=bot@example.edu
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=smtp_username
SMTP_PASS=smtp_password

4) Start server:  uvicorn main:app --reload --port 8000

Frontend CORS
-------------
Update allowed origins in the CORS section if your React app runs on a different origin (e.g., http://localhost:5173 or http://localhost:3000).

Notes
-----
- Dummy credentials are hardcoded below (STUDENT_CREDENTIALS / ADMIN_CREDENTIALS).
- Status values: 'not started', 'pending', 'done'. New complaints default to 'pending'.
- Public complaints shown on dashboards never include student identity.
- A lightweight keyword-based ragging classifier runs first, with Gemini API fallback if uncertain.
- If flagged, an SMTP email is sent to ANTI_RAGGING_EMAIL with complaint details.

requirements.txt (create this file alongside main.py)
----------------------------------------------------
fastapi==0.115.0
uvicorn==0.30.6
SQLAlchemy==2.0.34
pydantic==2.8.2
python-dotenv==1.0.1
google-generativeai==0.7.2
email-validator==2.2.0

"""

from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker, Mapped, mapped_column

# ------------------------------
# Environment & Config
# ------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
ANTI_RAGGING_EMAIL = os.getenv("ANTI_RAGGING_EMAIL")
FROM_EMAIL = os.getenv("FROM_EMAIL")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ------------------------------
# App / DB Setup
# ------------------------------
app = FastAPI(title="Agentic Campus Complaint Portal")

# Allow local dev React origins; add yours as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",  # relax for dev; tighten for prod
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base = declarative_base()
engine = create_engine("sqlite:///./complaints.db", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

# ------------------------------
# Models
# ------------------------------
class Student(Base):
    __tablename__ = "students"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(200), unique=True)
    password: Mapped[str] = mapped_column(String(128))  # plaintext for demo only

    complaints: Mapped[List[Complaint]] = relationship("Complaint", back_populates="student")


class Admin(Base):
    __tablename__ = "admins"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(200), unique=True)
    password: Mapped[str] = mapped_column(String(128))


class Complaint(Base):
    __tablename__ = "complaints"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id"))
    heading: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text())
    anonymous: Mapped[bool] = mapped_column(Boolean, default=False)
    public: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Agentic flags
    is_ragging_related: Mapped[bool] = mapped_column(Boolean, default=False)
    forwarded_to_arc: Mapped[bool] = mapped_column(Boolean, default=False)  # ARC = Anti Ragging Cell

    student: Mapped[Student] = relationship("Student", back_populates="complaints")

    __table_args__ = (
        CheckConstraint("status in ('not started','pending','done')", name="ck_status"),
    )


# ------------------------------
# Pydantic Schemas
# ------------------------------
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    role: str  # 'student' | 'admin'
    id: int
    name: str
    token: str = Field("demo-token", description="Dummy token for frontend state")

class ComplaintCreate(BaseModel):
    student_id: int
    heading: str
    description: str
    anonymous: bool = False
    public: bool = False

class ComplaintOut(BaseModel):
    id: int
    heading: str
    description: str
    anonymous: bool
    public: bool
    status: str
    created_at: datetime
    is_ragging_related: bool

class ComplaintOutWithStudent(ComplaintOut):
    student_id: int

class StatusUpdate(BaseModel):
    status: str = Field(pattern=r"^(not started|pending|done)$")

class DashboardStats(BaseModel):
    total: int
    resolved: int
    pending: int
    public_feed: List[ComplaintOut]

class ReportRequest(BaseModel):
    # Reserved for future parameters
    pass

class ReportResponse(BaseModel):
    report_text: str

# ------------------------------
# Dummy Credentials (Hardcoded for Demo)
# ------------------------------
STUDENT_CREDENTIALS = [
    {"id": 1, "name": "Alice Kumar", "email": "alice@univ.edu", "password": "alice123"},
    {"id": 2, "name": "Bala Singh", "email": "bala@univ.edu", "password": "bala123"},
]

ADMIN_CREDENTIALS = [
    {"id": 1, "name": "Dean Office", "email": "admin@univ.edu", "password": "admin123"},
]

# ------------------------------
# Helpers
# ------------------------------
RAGGING_KEYWORDS = {
    "ragging", "hazing", "harass", "harassment", "bully", "bullying",
    "assault", "threat", "extort", "freshers", "seniors forced",
}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_seed_data(db: Session) -> None:
    # Create tables
    Base.metadata.create_all(bind=engine)

    # Seed if empty
    if not db.scalar(select(func.count(Student.id))):
        for s in STUDENT_CREDENTIALS:
            db.add(Student(id=s["id"], name=s["name"], email=s["email"], password=s["password"]))
    if not db.scalar(select(func.count(Admin.id))):
        for a in ADMIN_CREDENTIALS:
            db.add(Admin(id=a["id"], name=a["name"], email=a["email"], password=a["password"]))

    if not db.scalar(select(func.count(Complaint.id))):
        # Some dummy complaints
        samples = [
            Complaint(student_id=1, heading="Water leakage in hostel", description="Bathroom on 2nd floor leaks.", anonymous=False, public=True, status="pending"),
            Complaint(student_id=2, heading="WiFi not working", description="Library WiFi down since yesterday.", anonymous=False, public=True, status="pending"),
            Complaint(student_id=1, heading="Seniors forcing introductions", description="Freshers asked to give intro loudly.", anonymous=True, public=False, status="pending"),
        ]
        db.add_all(samples)

    db.commit()


@app.on_event("startup")
def startup_event():
    with SessionLocal() as db:
        ensure_seed_data(db)


# ------------------------------
# Agentic Classifier & Email
# ------------------------------

def simple_keyword_classifier(text: str) -> Optional[bool]:
    if not text:
        return None
    lower = text.lower()
    if any(kw in lower for kw in RAGGING_KEYWORDS):
        return True
    return None  # uncertain


def gemini_is_ragging(text: str) -> Optional[bool]:
    if not GEMINI_API_KEY:
        return None
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a classifier for campus complaints.\n"
            "Task: Determine if the following complaint indicates ragging/hazing or harassment.\n"
            "Answer with a single token: YES or NO.\n\n"
            f"Complaint text:\n{text}\n"
        )
        resp = model.generate_content(prompt)
        content = (resp.text or "").strip().upper()
        if "YES" in content and "NO" not in content:
            return True
        if "NO" in content and "YES" not in content:
            return False
    except Exception as e:
        print("Gemini classification error:", e)
    return None


def send_ragging_email(complaint: Complaint):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and ANTI_RAGGING_EMAIL and FROM_EMAIL):
        print("[SMTP] Missing SMTP/.env config; skip sending.")
        return

    subject = f"[ARC FLAG] Complaint #{complaint.id}: {complaint.heading}"
    body = (
        f"Complaint ID: {complaint.id}\n"
        f"Created: {complaint.created_at} UTC\n"
        f"Student ID: {complaint.student_id} (anonymized={complaint.anonymous})\n"
        f"Public: {complaint.public}\n"
        f"Status: {complaint.status}\n\n"
        f"Description:\n{complaint.description}\n"
    )

    msg = MIMEMultipart()
    msg["From"] = FROM_EMAIL
    msg["To"] = ANTI_RAGGING_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        print(f"[SMTP] Sent ARC email for complaint {complaint.id}")
    except Exception as e:
        print("[SMTP] Error sending ARC email:", e)


# ------------------------------
# Endpoint Implementations
# ------------------------------

# Auth
@app.post("/login/student", response_model=LoginResponse)
def login_student(payload: LoginRequest, db: Session = Depends(get_db)):
    # Hardcoded auth
    s = next((x for x in STUDENT_CREDENTIALS if x["email"] == payload.email and x["password"] == payload.password), None)
    if not s:
        raise HTTPException(status_code=401, detail="Invalid student credentials")
    # Ensure presence in DB
    stu = db.scalar(select(Student).where(Student.email == s["email"]))
    if not stu:
        stu = Student(id=s["id"], name=s["name"], email=s["email"], password=s["password"])
        db.add(stu)
        db.commit()
    return LoginResponse(role="student", id=s["id"], name=s["name"]) 


@app.post("/login/admin", response_model=LoginResponse)
def login_admin(payload: LoginRequest, db: Session = Depends(get_db)):
    a = next((x for x in ADMIN_CREDENTIALS if x["email"] == payload.email and x["password"] == payload.password), None)
    if not a:
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    adm = db.scalar(select(Admin).where(Admin.email == a["email"]))
    if not adm:
        adm = Admin(id=a["id"], name=a["name"], email=a["email"], password=a["password"])
        db.add(adm)
        db.commit()
    return LoginResponse(role="admin", id=a["id"], name=a["name"]) 


# Student Dashboard
@app.get("/student/dashboard/{student_id}", response_model=DashboardStats)
def student_dashboard(student_id: int, db: Session = Depends(get_db)):
    # Stats for *all* complaints visible to student
    total = db.scalar(select(func.count(Complaint.id))) or 0
    resolved = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "done")) or 0
    pending = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "pending")) or 0

    # Recent public complaints (anonymized). You can add ORDER BY created_at DESC, LIMIT N in your frontend polling logic
    public_q = select(Complaint).where(Complaint.public == True).order_by(Complaint.created_at.desc()).limit(10)
    public_items = db.scalars(public_q).all()

    def to_out(c: Complaint) -> ComplaintOut:
        return ComplaintOut(
            id=c.id,
            heading=c.heading,
            description=c.description,
            anonymous=True,  # force anonymity on public feed
            public=c.public,
            status=c.status,
            created_at=c.created_at,
            is_ragging_related=c.is_ragging_related,
        )

    return DashboardStats(
        total=total,
        resolved=resolved,
        pending=pending,
        public_feed=[to_out(c) for c in public_items],
    )


# Submit Complaint
@app.post("/student/complaints", response_model=ComplaintOut)
def create_complaint(payload: ComplaintCreate, background: BackgroundTasks, db: Session = Depends(get_db)):
    # Validate student exists
    stu = db.get(Student, payload.student_id)
    if not stu:
        raise HTTPException(status_code=404, detail="Student not found")

    comp = Complaint(
        student_id=payload.student_id,
        heading=payload.heading.strip(),
        description=payload.description.strip(),
        anonymous=payload.anonymous,
        public=payload.public,
        status="pending",
    )
    db.add(comp)
    db.commit()
    db.refresh(comp)

    # Agentic classification
    text = f"{comp.heading}\n\n{comp.description}"
    decision = simple_keyword_classifier(text)
    if decision is None:
        decision = gemini_is_ragging(text)

    if decision is True:
        comp.is_ragging_related = True
        db.commit()
        # Send ARC email in background
        background.add_task(send_ragging_email, comp)

    return ComplaintOut(
        id=comp.id,
        heading=comp.heading,
        description=comp.description,
        anonymous=comp.anonymous,
        public=comp.public,
        status=comp.status,
        created_at=comp.created_at,
        is_ragging_related=comp.is_ragging_related,
    )


# My Complaints (by student)
@app.get("/student/complaints/{student_id}", response_model=List[ComplaintOut])
def my_complaints(student_id: int, db: Session = Depends(get_db)):
    # Ensure student exists
    if not db.get(Student, student_id):
        raise HTTPException(status_code=404, detail="Student not found")

    items = db.scalars(select(Complaint).where(Complaint.student_id == student_id).order_by(Complaint.created_at.desc())).all()
    return [
        ComplaintOut(
            id=c.id,
            heading=c.heading,
            description=c.description,
            anonymous=c.anonymous,
            public=c.public,
            status=c.status,
            created_at=c.created_at,
            is_ragging_related=c.is_ragging_related,
        ) for c in items
    ]


# Admin Dashboard
@app.get("/admin/dashboard", response_model=DashboardStats)
def admin_dashboard(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Complaint.id))) or 0
    resolved = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "done")) or 0
    pending = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "pending")) or 0

    public_q = select(Complaint).where(Complaint.public == True).order_by(Complaint.created_at.desc()).limit(10)
    public_items = db.scalars(public_q).all()

    def to_out(c: Complaint) -> ComplaintOut:
        return ComplaintOut(
            id=c.id,
            heading=c.heading,
            description=c.description,
            anonymous=True,  # still anonymized on public feed
            public=c.public,
            status=c.status,
            created_at=c.created_at,
            is_ragging_related=c.is_ragging_related,
        )

    return DashboardStats(
        total=total,
        resolved=resolved,
        pending=pending,
        public_feed=[to_out(c) for c in public_items],
    )


# Admin: List All Complaints
@app.get("/admin/complaints", response_model=List[ComplaintOutWithStudent])
def list_all_complaints(db: Session = Depends(get_db)):
    items = db.scalars(select(Complaint).order_by(Complaint.created_at.desc())).all()
    return [
        ComplaintOutWithStudent(
            id=c.id,
            student_id=c.student_id,
            heading=c.heading,
            description=c.description,
            anonymous=c.anonymous,
            public=c.public,
            status=c.status,
            created_at=c.created_at,
            is_ragging_related=c.is_ragging_related,
        ) for c in items
    ]


# Admin: Update Complaint Status
@app.put("/admin/complaints/{complaint_id}", response_model=ComplaintOutWithStudent)
def update_status(complaint_id: int, payload: StatusUpdate, db: Session = Depends(get_db)):
    comp = db.get(Complaint, complaint_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Complaint not found")

    comp.status = payload.status
    db.commit()
    db.refresh(comp)

    return ComplaintOutWithStudent(
        id=comp.id,
        student_id=comp.student_id,
        heading=comp.heading,
        description=comp.description,
        anonymous=comp.anonymous,
        public=comp.public,
        status=comp.status,
        created_at=comp.created_at,
        is_ragging_related=comp.is_ragging_related,
    )


# Admin: Generate Report with Gemini
@app.post("/admin/report", response_model=ReportResponse)
def generate_report(_: ReportRequest | None = None, db: Session = Depends(get_db)):
    # Gather a compact snapshot of recent complaints & stats
    total = db.scalar(select(func.count(Complaint.id))) or 0
    resolved = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "done")) or 0
    pending = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "pending")) or 0

    recent_items = db.scalars(
        select(Complaint).order_by(Complaint.created_at.desc()).limit(50)
    ).all()

    lines = [
        f"Total={total}",
        f"Resolved(done)={resolved}",
        f"Pending={pending}",
        "Recent complaints (latest 50):",
    ]
    for c in recent_items:
        lines.append(
            f"- #{c.id} [{c.status}] ragging={c.is_ragging_related} public={c.public} anon={c.anonymous} :: {c.heading} :: {c.description[:180]}"
        )
    snapshot = "\n".join(lines)

    if not GEMINI_API_KEY:
        # Fallback: simple text if Gemini not configured
        return ReportResponse(report_text=("Gemini API key missing. Showing raw snapshot.\n\n" + snapshot))

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a campus operations analyst.\n"
            "Given complaint stats and a list of recent complaints, produce a concise report with:\n"
            "1) Key metrics and trends, 2) Top 3 categories of issues, 3) Actionable recommendations (bulleted), 4) Any ragging-related red flags to escalate.\n"
            "Keep it under 300 words.\n\n"
            f"DATA:\n{snapshot}\n"
        )
        resp = model.generate_content(prompt)
        report = resp.text or "(No content)"
        return ReportResponse(report_text=report.strip())
    except Exception as e:
        return ReportResponse(report_text=f"Gemini error: {e}\n\nRaw snapshot:\n{snapshot}")


# Healthcheck (optional)
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}
