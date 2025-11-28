from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import os

from db import init_db, db
from models import Candidate, Job, Match

# Disable tokenizer parallelism warnings and reduce overhead
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------
# CONFIG & MODEL LOADING
# -------------------------
app = Flask(__name__)
init_db(app)

with app.app_context():
    db.create_all()

# Use a lighter model to fit in 512MB RAM on free hosting
# (still good quality for semantic similarity)
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

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

    if request.method == "POST":
        resume_text = request.form.get("resume_text", "")
        job_text = request.form.get("job_text", "")
        candidate_name = request.form.get("candidate_name", "Anonymous")
        candidate_email = request.form.get("candidate_email", "")

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

    return render_template("index.html", result=result, matches=matches)


# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == "__main__":
    # For Render / other PaaS: use PORT env, bind to 0.0.0.0
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
