import streamlit as st


def render_progress(items: dict[str, int]) -> None:
    for label, percentage in items.items():
        st.write(f"**{label}** - {percentage}%")
        st.progress(max(0, min(100, percentage)) / 100)
