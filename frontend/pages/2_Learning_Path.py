import streamlit as st

from components.roadmap import render_roadmap

st.set_page_config(page_title="Learning Path", page_icon="🗺️", layout="wide")
st.title("Learning Path")
st.caption("A suggested sequence of milestones for your next stage of growth.")

render_roadmap([
    ("Programming foundations", "Build confidence with core programming concepts.", "Complete"),
    ("Database development", "Practice relational modeling and SQL queries.", "In progress"),
    ("Application project", "Apply your skills in a portfolio-ready project.", "Upcoming"),
])
