from __future__ import annotations
import os
import streamlit as st

def inject_css(project_root: str):
    css_path = os.path.join(project_root, "assets", "styles.css")
    if os.path.exists(css_path):
        st.markdown(f"<style>{open(css_path, 'r', encoding='utf-8').read()}</style>", unsafe_allow_html=True)

def sidebar_branding(*, title: str, subtitle: str, badges: list[str] | None = None):
    with st.sidebar:
        st.markdown(
            f"""
            <div class="card" style="margin-top:8px;">
              <div class="card-title">{title}</div>
              <div class="card-value" style="font-size:1.1rem;">{subtitle}</div>
              {"".join([f"<div style='opacity:.75; margin-top:6px;'>{b}</div>" for b in (badges or [])])}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div class='section'></div>", unsafe_allow_html=True)

def header(title: str, caption: str):
    st.markdown(
        f"""
        <div style="margin-bottom: 10px;">
          <h2 style="margin-bottom: 4px;">{title}</h2>
          <div style="opacity: 0.75;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
