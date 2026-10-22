from collections.abc import Iterable

import streamlit as st


def render_roadmap(items: Iterable[tuple[str, str, str]]) -> None:
    for title, description, status in items:
        with st.container(border=True):
            st.subheader(title)
            st.write(description)
            st.caption(f"Status: {status}")
