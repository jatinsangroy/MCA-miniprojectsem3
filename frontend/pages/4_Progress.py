import streamlit as st

from components.progress import render_progress

st.set_page_config(page_title="Progress", page_icon="📈", layout="wide")
st.title("Your Progress")
st.caption("Keep an eye on the milestones you have completed.")

render_progress({
    "Programming foundations": 100,
    "Database development": 60,
    "Application project": 25,
})
