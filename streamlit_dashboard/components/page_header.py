"""
page_header.py — 페이지 상단 핵심 메시지 박스
=================================================
C-Level 보고용. 각 페이지가 무엇을 보여주는지 1~2줄로 요약.
"""
from __future__ import annotations

import streamlit as st


def render_page_intro(message: str, accent: str = "#003087") -> None:
    """페이지 상단 'so what' 박스.

    Args:
        message: 페이지 핵심 메시지 (1~2줄, HTML 허용)
        accent:  좌측 accent bar 색상 (기본 FILA 네이비)
    """
    st.markdown(
        f"""<div style='background:#F4F7FE; border-left:4px solid {accent};
                    padding:12px 18px; margin:6px 0 18px;
                    border-radius:0 4px 4px 0;'>
            <div style='font-weight:600; font-size:11px; color:{accent};
                        letter-spacing:0.5px; margin-bottom:4px;'>
                이 페이지에서 알 수 있는 것
            </div>
            <div style='font-size:14px; line-height:1.5; color:#222;'>
                {message}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
