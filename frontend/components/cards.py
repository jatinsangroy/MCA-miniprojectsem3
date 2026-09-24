import streamlit as st


def render_section_title(title: str, description: str = "") -> None:
    st.title(title)
    if description:
        st.caption(description)


def render_stat_card(label: str, value: str, description: str = "") -> None:
    st.metric(label, value, help=description or None)
