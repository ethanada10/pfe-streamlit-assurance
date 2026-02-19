import streamlit as st

def metric_card(title: str, value: str, subtitle: str | None = None):
    html = f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      {f'<div style="opacity:0.75; margin-top:6px;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
