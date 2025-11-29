from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from PyPDF2 import PdfReader

from db import init_db, db
from models import Candidate, Job, Match

# -------------------------
# CONFIG & MODEL LOADING
# -------------------------
app = Flask(__name__)
init_db(app)

with app.app_context():
    db.create_all()

# Use a MiniLM model for semantic similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

# A simple global skill vocabulary – extend this as needed
SKILL_VOCAB = [
    # Programming languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "go", "rust", "kotlin", "swift", "php", "ruby",

    # Web technologies
    "html", "css", "react", "angular", "vue", "nextjs", "nodejs",
    "express", "django", "flask", "spring boot",

    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis",

    # Data / AI / ML
    "machine learning", "deep learning", "nlp", "data analysis",
    "pandas", "numpy", "pytorch", "tensorflow", "computer vision",

    # DevOps / Cloud
    "git", "github", "gitlab", "docker", "kubernetes",
    "aws", "azure", "gcp", "ci/cd",

    # Analytics / BI
    "power bi", "tableau", "excel",

    # General / soft skills
    "problem solving", "communication", "leadership",
    "teamwork", "time management", "presentation skills"
]


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def extract_text_from_pdf(file_stream) -> str:
    """Extract plain text from an uploaded PDF file."""
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += "\n" + page_text
        return text.strip()
    except Exception as e:
        print("PDF read error:", e)
        return ""


def embed(text: str) -> np.ndarray:
    if not text:
        # embedding size for MiniLM models is 384
        return np.zeros((1, 384))
    emb = model.encode([text])
    return np.array(emb)


def compute_match_score(resume_text: str, job_text: str) -> float:
    """Return similarity in percentage (0–100)."""
    e1 = embed(resume_text)
    e2 = embed(job_text)
    sim = cosine_similarity(e1, e2)[0][0]
    # convert to 0–100 range
    score = float(max(0.0, min(1.0, (sim + 1) / 2))) * 100
    return round(score, 2)


def extract_skills(text: str):
    """Very simple skill extraction based on vocabulary match."""
    text_lower = text.lower()
    found = []
    for s in SKILL_VOCAB:
        if s in text_lower:
            found.append(s)
    return sorted(list(set(found)))


def compute_skill_gap(candidate_skills, job_skills):
    candidate_set = {s.lower() for s in candidate_skills}
    job_set = {s.lower() for s in job_skills}
    missing = list(job_set - candidate_set)
    return sorted(missing)


# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error_message = None

    if request.method == "POST":
        candidate_name = request.form.get("candidate_name", "Anonymous")
        candidate_email = request.form.get("candidate_email", "")
        job_text = request.form.get("job_text", "").strip()

        # --- PDF upload is REQUIRED for resume ---
        pdf_file = request.files.get("resume_pdf")
        resume_text = ""

        if pdf_file and pdf_file.filename:
            if pdf_file.mimetype == "application/pdf":
                resume_text = extract_text_from_pdf(pdf_file.stream)
                if not resume_text:
                    error_message = "Could not read text from the uploaded PDF."
            else:
                error_message = "Please upload a valid PDF file for the resume."
        else:
            error_message = "Please upload your resume as a PDF file."

        if not error_message:
            # 1. Extract skills
            cand_skills = extract_skills(resume_text)
            job_skills = extract_skills(job_text)

            # 2. Semantic similarity
            match_score = compute_match_score(resume_text, job_text)

            # 3. Skill gap
            missing_skills = compute_skill_gap(cand_skills, job_skills)

            # 4. Store in DB
            candidate = Candidate(
                name=candidate_name,
                email=candidate_email,
                resume_text=resume_text,
                skills=",".join(cand_skills),
            )
            job = Job(
                title="Hackathon Job Role",
                description=job_text,
                skills_required=",".join(job_skills),
            )
            db.session.add(candidate)
            db.session.add(job)
            db.session.commit()

            m = Match(
                candidate_id=candidate.id,
                job_id=job.id,
                match_score=match_score,
                missing_skills=",".join(missing_skills),
            )
            db.session.add(m)
            db.session.commit()

            result = {
                "match_score": match_score,
                "cand_skills": cand_skills,
                "job_skills": job_skills,
                "missing_skills": missing_skills,
                "candidate": candidate,
                "job": job,
            }

    # For recruiter view – show top candidates
    matches = (
        Match.query.order_by(Match.match_score.desc())
        .limit(10)
        .all()
    )

    return render_template("index.html", result=result, matches=matches, error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True)
