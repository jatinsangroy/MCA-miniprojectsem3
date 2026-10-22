import streamlit as st

from utils.helpers import calculate_score

st.set_page_config(page_title="Assessment", page_icon="📝", layout="wide")
st.title("Skill Assessment")
st.caption("Check your understanding and identify your next focus area.")

questions = [
    ("Which data structure follows FIFO order?", ["Stack", "Queue", "Tree"], "Queue"),
    ("Which language is commonly used for data analysis?", ["Python", "HTML", "CSS"], "Python"),
]

with st.form("assessment_form"):
    answers = [st.radio(question, options) for question, options, _ in questions]
    submitted = st.form_submit_button("Submit assessment")

if submitted:
    score = calculate_score(answers, [answer for _, _, answer in questions])
    st.metric("Score", f"{score}/{len(questions)}")
