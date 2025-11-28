# models.py
from datetime import datetime
from db import db

class Candidate(db.Model):
    __tablename__ = "candidates"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120))
    resume_text = db.Column(db.Text)
    skills = db.Column(db.Text)  # comma separated skills
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def skill_list(self):
        return [s.strip() for s in (self.skills or "").split(",") if s.strip()]


class Job(db.Model):
    __tablename__ = "jobs"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200))
    description = db.Column(db.Text)
    skills_required = db.Column(db.Text)  # comma separated
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def skill_list(self):
        return [s.strip() for s in (self.skills_required or "").split(",") if s.strip()]


class Match(db.Model):
    __tablename__ = "matches"

    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidates.id"))
    job_id = db.Column(db.Integer, db.ForeignKey("jobs.id"))
    match_score = db.Column(db.Float)
    missing_skills = db.Column(db.Text)  # comma separated

    candidate = db.relationship("Candidate", backref="matches")
    job = db.relationship("Job", backref="matches")
