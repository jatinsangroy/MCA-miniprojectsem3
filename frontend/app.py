import streamlit as st

from components.navbar import render_navbar

st.set_page_config(
    page_title="Learning Path",
    page_icon="🎓",
    layout="wide",
)

with open("frontend/assets/style.css", encoding="utf-8") as stylesheet:
    st.markdown(f"<style>{stylesheet.read()}</style>", unsafe_allow_html=True)

render_navbar()

st.title("Personalized Learning Path")
st.write("Build skills, track progress, and stay focused on your MCA journey.")

left, right = st.columns(2)
with left:
    st.subheader("Welcome back")
    st.info("Complete your profile to generate a learning path tailored to your goals.")
with right:
    st.subheader("Getting started")
    st.markdown("1. Create your profile\n2. Explore your learning path\n3. Take an assessment\n4. Review your progress")
