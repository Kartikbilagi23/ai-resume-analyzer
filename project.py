from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model=SentenceTransformer("all-MiniLM-L6-v2")

def calculate_sim(resume_text,jd_text):
    emb=model.encode([resume_text,jd_text])
    sim=cosine_similarity([emb[0],emb[1]])
    return round(sim[0][0]*100,2)

SKILLS = [
    "python", "java", "c++", "machine learning", "deep learning",
    "nlp", "data science", "sql", "mongodb", "react", "node",
    "fastapi", "flask", "aws", "docker", "git"
]
def extract_skills(text):
    text = text.lower()
    found = [skill for skill in SKILLS if skill in text]
    return set(found)

def find_missing_skills(resume,jd):
    resume_skill=extract_skills(resume)
    jd_skills=extract_skills(jd)
    missing=jd_skills-resume_skill
    return missing

import streamlit as st
st.title("📄 AI Resume Analyzer")
resume_text=st.text_area("Paste Your Resume here")
jd_text=st.text_area("Paste Job Description here")
if st.button('Analyse'):
    if resume_text and jd_text:
        score=calculate_sim(resume_text,jd_text)
        missing=find_missing_skills(resume_text,jd_text)

        st.subheader("🔍 Match Score")
        st.success(f"{score}%")

        st.subheader("❌ Missing Skills")
        if missing:
            st.write(", ".join(missing))
        else:
            st.write("No major skills missing 🎉")
    else:
        st.warning("Please paste both Resume and Job Description")

