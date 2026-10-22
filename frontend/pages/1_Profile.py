import streamlit as st

from components.cards import render_section_title

st.set_page_config(page_title="Profile", page_icon="👤", layout="wide")
render_section_title("Your Profile", "Tell us about your goals and current skills.")

with st.form("profile_form"):
    name = st.text_input("Name")
    goal = st.selectbox("Primary goal", ["Software development", "Data science", "Cloud computing", "Cybersecurity"])
    skills = st.text_area("Current skills", placeholder="Python, SQL, Git")
    submitted = st.form_submit_button("Save profile")

if submitted:
    st.success(f"Profile saved for {name or 'learner'}.")
