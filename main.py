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
- Status values: 'pending', 'in_progress', 'resolved'. New complaints default to 'pending'.
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
import logging
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Response
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
# Logging (visible on your host logs, e.g. Railway)
# ------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("complaints")

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
    log.info("Gemini configured: key present")
else:
    log.info("Gemini not configured: GEMINI_API_KEY missing")

# ------------------------------
# App / DB Setup
# ------------------------------
app = FastAPI(title="Agentic Campus Complaint Portal")

# CORS: permissive for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.options("/{rest_of_path:path}")
def cors_preflight(rest_of_path: str):
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

Base = declarative_base()

# SQLite connection for multi-threaded FastAPI
engine = create_engine(
    "sqlite:///./complaints.db",
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
)
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

    complaints: Mapped[List["Complaint"]] = relationship("Complaint", back_populates="student")


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

    student: Mapped["Student"] = relationship("Student", back_populates="complaints")

    __table_args__ = (
        CheckConstraint("status in ('pending','in_progress','resolved')", name="ck_status"),
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
    status: str = Field(pattern=r"^(pending|in_progress|resolved)$")

class DashboardStats(BaseModel):
    total: int
    resolved: int
    pending: int
    public_feed: List[ComplaintOut]

class ReportRequest(BaseModel):
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
    "force introduction", "forced introduction", "physical abuse",
    "verbal abuse", "intimidate", "coerce", "rag", "ragged"
}

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _needs_status_migration(conn) -> bool:
    cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='complaints'")
    row = cur.fetchone()
    if not row or not row[0]:
        return False
    ddl = row[0].lower()
    return ("not started" in ddl) or ("done" in ddl)

def _migrate_complaints_table(conn) -> None:
    conn.executescript("""
    PRAGMA foreign_keys=off;
    BEGIN TRANSACTION;

    CREATE TABLE IF NOT EXISTS complaints_new (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER NOT NULL,
        heading VARCHAR(200) NOT NULL,
        description TEXT NOT NULL,
        anonymous BOOLEAN NOT NULL DEFAULT 0,
        public BOOLEAN NOT NULL DEFAULT 0,
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        is_ragging_related BOOLEAN NOT NULL DEFAULT 0,
        forwarded_to_arc BOOLEAN NOT NULL DEFAULT 0,
        FOREIGN KEY(student_id) REFERENCES students(id)
    );

    INSERT INTO complaints_new (id, student_id, heading, description, anonymous, public, status, created_at, is_ragging_related, forwarded_to_arc)
    SELECT
        id,
        student_id,
        heading,
        description,
        COALESCE(anonymous, 0),
        COALESCE(public, 0),
        CASE
            WHEN LOWER(status) IN ('resolved','done') THEN 'resolved'
            WHEN LOWER(status) IN ('in_progress','in progress','in-progress') THEN 'in_progress'
            ELSE 'pending'
        END AS status,
        created_at,
        COALESCE(is_ragging_related, 0),
        COALESCE(forwarded_to_arc, 0)
    FROM complaints;

    DROP TABLE complaints;
    ALTER TABLE complaints_new RENAME TO complaints;

    COMMIT;
    PRAGMA foreign_keys=on;
    """)

def ensure_seed_data(db: Session) -> None:
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        try:
            if _needs_status_migration(conn.connection):
                log.info("Running complaints table migration (status normalization)")
                _migrate_complaints_table(conn.connection)
        except Exception as e:
            log.error("Migration skipped due to error: %s", e)

    if not db.scalar(select(func.count(Student.id))):
        for s in STUDENT_CREDENTIALS:
            db.add(Student(id=s["id"], name=s["name"], email=s["email"], password=s["password"]))
    if not db.scalar(select(func.count(Admin.id))):
        for a in ADMIN_CREDENTIALS:
            db.add(Admin(id=a["id"], name=a["name"], email=a["email"], password=a["password"]))

    if not db.scalar(select(func.count(Complaint.id))):
        samples = [
            Complaint(student_id=1, heading="Water leakage in hostel", description="Bathroom on 2nd floor leaks.", anonymous=False, public=True, status="resolved"),
            Complaint(student_id=2, heading="WiFi not working", description="Library WiFi down since yesterday.", anonymous=False, public=True, status="pending"),
            Complaint(student_id=1, heading="Seniors forcing introductions", description="Freshers asked to give intro loudly.", anonymous=True, public=False, status="pending"),
        ]
        db.add_all(samples)
    db.commit()

@app.on_event("startup")
def startup_event():
    with SessionLocal() as db:
        ensure_seed_data(db)
    log.info("Startup complete. SMTP_HOST=%s, ANTI_RAGGING_EMAIL=%s, FROM_EMAIL=%s",
             SMTP_HOST, ANTI_RAGGING_EMAIL, FROM_EMAIL)

# ------------------------------
# Classifier & Email
# ------------------------------
def simple_keyword_classifier(text: str) -> Optional[bool]:
    if not text:
        return None
    lower = text.lower()
    hit = any(kw in lower for kw in RAGGING_KEYWORDS)
    log.info("[Classifier] keyword=%s", hit)
    if hit:
        return True
    return None

def gemini_is_ragging(text: str) -> Optional[bool]:
    if not GEMINI_API_KEY:
        log.info("[Classifier] Gemini not configured; skip")
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
        decision = None
        if "YES" in content and "NO" not in content:
            decision = True
        elif "NO" in content and "YES" not in content:
            decision = False
        log.info("[Classifier] Gemini decision=%s (raw=%s)", decision, content)
        return decision
    except Exception as e:
        log.error("Gemini classification error: %s", e)
    return None

def _ensure_outbox_dir() -> str:
    out_dir = os.path.join(os.getcwd(), "outbox")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _save_eml_copy(filename_base: str, msg: MIMEMultipart) -> None:
    out_dir = _ensure_outbox_dir()
    path = os.path.join(out_dir, f"{filename_base}.eml")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(msg.as_string())
        log.info("[OUTBOX] Saved email copy to %s", path)
    except Exception as e:
        log.error("[OUTBOX] Failed to save .eml copy: %s", e)

def send_ragging_email(complaint: Complaint):
    subject = f"[ARC FLAG] Complaint #{complaint.id}: {complaint.heading}"
    body = (
        f"Complaint ID: {complaint.id}\n"
        f"Created: {complaint.created_at} UTC\n"
        f"Student ID: {complaint.student_id} (anonymized={complaint.anonymous})\n"
        f"Public: {complaint.public}\n"
        f"Status: {complaint.status}\n\n"
        f"Description:\n{complaint.description}\n"
    )

    to_addr = ANTI_RAGGING_EMAIL or "antiraggingcell@localhost"
    from_addr = FROM_EMAIL or "noreply@localhost"

    msg = MIMEMultipart()
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["X-Auto-Generated"] = "Yes"
    msg.attach(MIMEText(body, "plain"))

    _save_eml_copy(f"arc-{complaint.id}", msg)

    if not SMTP_HOST:
        log.warning("[SMTP] SMTP_HOST missing; wrote .eml only (no network send)")
        return

    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                if SMTP_USER and SMTP_PASS:
                    server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.ehlo()
                try:
                    server.starttls()
                    server.ehlo()
                except Exception as e:
                    log.warning("[SMTP] STARTTLS not available/failed: %s (sending without TLS)", e)
                if SMTP_USER and SMTP_PASS:
                    server.login(SMTP_USER, SMTP_PASS)
                server.send_message(msg)
        log.info("[SMTP] Sent ARC email for complaint %s", complaint.id)
    except Exception as e:
        log.error("[SMTP] Error sending ARC email: %s", e)
        # .eml already saved

# ------------------------------
# Endpoint Implementations
# ------------------------------

# Auth
@app.post("/login/student", response_model=LoginResponse)
def login_student(payload: LoginRequest, db: Session = Depends(get_db)):
    s = next((x for x in STUDENT_CREDENTIALS if x["email"] == payload.email and x["password"] == payload.password), None)
    if not s:
        raise HTTPException(status_code=401, detail="Invalid student credentials")
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
    total = db.scalar(select(func.count(Complaint.id))) or 0
    resolved = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "resolved")) or 0
    pending = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "pending")) or 0

    public_q = select(Complaint).where(Complaint.public == True).order_by(Complaint.created_at.desc()).limit(10)
    public_items = db.scalars(public_q).all()

    def to_out(c: Complaint) -> ComplaintOut:
        return ComplaintOut(
            id=c.id, heading=c.heading, description=c.description,
            anonymous=True, public=c.public, status=c.status,
            created_at=c.created_at, is_ragging_related=c.is_ragging_related,
        )

    return DashboardStats(
        total=total, resolved=resolved, pending=pending,
        public_feed=[to_out(c) for c in public_items],
    )

# Submit Complaint
@app.post("/student/complaints", response_model=ComplaintOut)
def create_complaint(payload: ComplaintCreate, background: BackgroundTasks, db: Session = Depends(get_db)):
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
    log.info("[Create] Complaint #%s created (public=%s, anonymous=%s)", comp.id, comp.public, comp.anonymous)

    text = f"{comp.heading}\n\n{comp.description}"
    decision = simple_keyword_classifier(text)
    if decision is None:
        decision = gemini_is_ragging(text)

    log.info("[Create] ragging_decision=%s for complaint #%s", decision, comp.id)

    if decision is True:
        comp.is_ragging_related = True
        comp.forwarded_to_arc = True
        db.commit()
        log.info("[Create] Complaint #%s flagged for ARC (queued email)", comp.id)
        background.add_task(send_ragging_email, comp)

    return ComplaintOut(
        id=comp.id, heading=comp.heading, description=comp.description,
        anonymous=comp.anonymous, public=comp.public, status=comp.status,
        created_at=comp.created_at, is_ragging_related=comp.is_ragging_related,
    )

# Recent/Public complaints endpoint
@app.get("/student/complaints", response_model=List[ComplaintOut])
def list_complaints(public: bool = Query(False), limit: int = Query(10, ge=1, le=100), db: Session = Depends(get_db)):
    q = select(Complaint)
    if public:
        q = q.where(Complaint.public == True)
    q = q.order_by(Complaint.created_at.desc()).limit(limit)
    items = db.scalars(q).all()
    return [
        ComplaintOut(
            id=c.id, heading=c.heading, description=c.description,
            anonymous=c.anonymous if not public else True,
            public=c.public, status=c.status, created_at=c.created_at,
            is_ragging_related=c.is_ragging_related,
        ) for c in items
    ]

# My Complaints
@app.get("/student/complaints/{student_id}", response_model=List[ComplaintOut])
def my_complaints(student_id: int, db: Session = Depends(get_db)):
    if not db.get(Student, student_id):
        raise HTTPException(status_code=404, detail="Student not found")
    items = db.scalars(select(Complaint).where(Complaint.student_id == student_id).order_by(Complaint.created_at.desc())).all()
    return [
        ComplaintOut(
            id=c.id, heading=c.heading, description=c.description,
            anonymous=c.anonymous, public=c.public, status=c.status,
            created_at=c.created_at, is_ragging_related=c.is_ragging_related,
        ) for c in items
    ]

# Admin Dashboard
@app.get("/admin/dashboard", response_model=DashboardStats)
def admin_dashboard(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Complaint.id))) or 0
    resolved = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "resolved")) or 0
    pending = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "pending")) or 0

    public_q = select(Complaint).where(Complaint.public == True).order_by(Complaint.created_at.desc()).limit(10)
    public_items = db.scalars(public_q).all()

    def to_out(c: Complaint) -> ComplaintOut:
        return ComplaintOut(
            id=c.id, heading=c.heading, description=c.description,
            anonymous=True, public=c.public, status=c.status,
            created_at=c.created_at, is_ragging_related=c.is_ragging_related,
        )

    return DashboardStats(
        total=total, resolved=resolved, pending=pending,
        public_feed=[to_out(c) for c in public_items],
    )

# Admin: List All Complaints
@app.get("/admin/complaints", response_model=List[ComplaintOutWithStudent])
def list_all_complaints(db: Session = Depends(get_db)):
    items = db.scalars(select(Complaint).order_by(Complaint.created_at.desc())).all()
    return [
        ComplaintOutWithStudent(
            id=c.id, student_id=c.student_id, heading=c.heading, description=c.description,
            anonymous=c.anonymous, public=c.public, status=c.status,
            created_at=c.created_at, is_ragging_related=c.is_ragging_related,
        ) for c in items
    ]

# Admin: Update Complaint Status
@app.put("/admin/complaints/{complaint_id}", response_model=ComplaintOutWithStudent)
def update_status(complaint_id: int, payload: StatusUpdate, db: Session = Depends(get_db)):
    comp = db.get(Complaint, complaint_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Complaint not found")
    prev = comp.status
    comp.status = payload.status
    db.commit()
    db.refresh(comp)
    log.info("[Status] #%s: %s -> %s", comp.id, prev, comp.status)
    return ComplaintOutWithStudent(
        id=comp.id, student_id=comp.student_id, heading=comp.heading, description=comp.description,
        anonymous=comp.anonymous, public=comp.public, status=comp.status,
        created_at=comp.created_at, is_ragging_related=comp.is_ragging_related,
    )

# Admin: Generate Report with Gemini
@app.post("/admin/report", response_model=ReportResponse)
def generate_report(_: ReportRequest | None = None, db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Complaint.id))) or 0
    resolved = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "resolved")) or 0
    pending = db.scalar(select(func.count(Complaint.id)).where(Complaint.status == "pending")) or 0

    recent_items = db.scalars(select(Complaint).order_by(Complaint.created_at.desc()).limit(50)).all()

    lines = [
        f"Total={total}",
        f"Resolved(resolved)={resolved}",
        f"Pending={pending}",
        "Recent complaints (latest 50):",
    ]
    for c in recent_items:
        lines.append(
            f"- #{c.id} [{c.status}] ragging={c.is_ragging_related} public={c.public} anon={c.anonymous} :: {c.heading} :: {c.description[:180]}"
        )
    snapshot = "\n".join(lines)

    if not GEMINI_API_KEY:
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
        log.error("Gemini error: %s", e)
        return ReportResponse(report_text=f"Gemini error: {e}\n\nRaw snapshot:\n{snapshot}")

# Healthcheck
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# ------------------------------
# Debug / Self-test endpoints (non-breaking, optional)
# ------------------------------
class DebugText(BaseModel):
    text: str

@app.get("/debug/ping")
def debug_ping():
    return {
        "ok": True,
        "smtp_host": SMTP_HOST or "(missing)",
        "smtp_port": SMTP_PORT,
        "smtp_user_set": bool(SMTP_USER),
        "from_email": FROM_EMAIL or "(missing)",
        "to_email": ANTI_RAGGING_EMAIL or "(missing)",
        "gemini_configured": bool(GEMINI_API_KEY),
        "outbox_dir": os.path.abspath("outbox"),
        "time": datetime.utcnow().isoformat() + "Z",
    }

@app.post("/debug/classify")
def debug_classify(payload: DebugText):
    kw = simple_keyword_classifier(payload.text)
    gm = None if kw is not None else gemini_is_ragging(payload.text)
    decision = kw if kw is not None else gm
    return {
        "keyword_decision": kw,
        "gemini_decision": gm,
        "final_decision": decision
    }

@app.post("/debug/test_email")
def debug_test_email():
    # Compose a fake complaint object in-memory
    fake = Complaint(
        id=999999,
        student_id=0,
        heading="TEST: Ragging alert",
        description="This is a test email from /debug/test_email endpoint.",
        anonymous=True,
        public=False,
        status="pending",
        created_at=datetime.utcnow(),
        is_ragging_related=True,
        forwarded_to_arc=True,
    )
    log.info("[Debug] Sending test ARC email (id=%s)", fake.id)
    try:
        send_ragging_email(fake)
        return {"ok": True, "message": "Test email attempt finished. Check logs and ./outbox/."}
    except Exception as e:
        log.error("[Debug] Test email error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
