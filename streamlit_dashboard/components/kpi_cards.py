"""
kpi_cards.py — KPI 카드 컴포넌트
=================================
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import BRANDS, BRAND_ORDER


def kpi_row(kpi_df: pd.DataFrame) -> None:
    """브랜드별 KPI 카드 4개 가로 배치."""
    if kpi_df.empty:
        st.info("KPI 산출 데이터 없음")
        return

    cols = st.columns(len(BRAND_ORDER))
    for col, brand in zip(cols, BRAND_ORDER):
        row = kpi_df[kpi_df["brand"] == brand]
        if row.empty:
            with col:
                _empty_card(brand)
            continue
        r = row.iloc[0]
        with col:
            _brand_card(
                brand=brand,
                n_reviews=int(r.get("n_reviews", 0)),
                mean_rating=float(r.get("mean_rating", 0.0)),
                rating_std=float(r.get("rating_std", 0.0)),
            )


def _brand_card(brand: str, n_reviews: int, mean_rating: float, rating_std: float):
    meta = BRANDS[brand]
    color = meta["color"]
    is_self = meta["is_self"]
    badge = "<span style='background:#FFD700;color:#000;padding:2px 6px;border-radius:4px;font-size:11px;'>자사</span>" if is_self else ""
    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; padding: 10px 14px; background:#1e1e2e15; border-radius: 4px;">
            <div style="font-size:12px; color:#888;">{badge} {brand}</div>
            <div style="font-size:13px; color:#ccc; margin-top:2px;">{meta['label']}</div>
            <div style="font-size:24px; font-weight:600; margin-top:6px;">{n_reviews:,}</div>
            <div style="font-size:11px; color:#888;">리뷰 수</div>
            <div style="margin-top:8px; font-size:14px;"> {mean_rating:.2f} <span style="color:#888; font-size:11px;">±{rating_std:.2f}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _empty_card(brand: str):
    st.markdown(
        f"""<div style="border-left: 4px dashed #666; padding: 10px 14px; opacity:0.5;">
        <div style="font-size:12px;">{brand}</div>
        <div style="font-size:14px; color:#888;">데이터 없음</div></div>""",
        unsafe_allow_html=True,
    )


def metric_grid(metrics: list[dict], cols: int = 4) -> None:
    """일반 metric 카드 그리드. metrics=[{label,value,delta,help}]"""
    columns = st.columns(cols)
    for i, m in enumerate(metrics):
        with columns[i % cols]:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                help=m.get("help"),
            )
