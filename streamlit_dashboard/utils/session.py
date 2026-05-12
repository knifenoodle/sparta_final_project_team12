"""
session.py — 전역 상태 관리 (Session State)
============================================

페이지 전환에도 유지되는 필터/선택 상태의 단일 저장소.

원칙:
- 모든 페이지는 init_session() 을 페이지 진입 시 1회 호출.
- get_filters() 는 dict 복사본 반환 (외부 변형 방지).
"""
from __future__ import annotations

import streamlit as st

from config import BRAND_ORDER

# 기본 필터값
_DEFAULTS = {
    "brands":        list(BRAND_ORDER),  # list[str] — 사이드바 multi pills 호환
    "rating_sel":    "전체",            # selectbox → str ("전체" / "1점" … "5점")
    "cat1_filters":  [],                # checkbox → list[str], 빈 리스트 = 전체
    "cat2_filters":  [],
    "cat3_filters":  [],
    "year_range":    (2024, 2026),      # filters.py 슬라이더 범위와 동기화
    "price_range":   (0, 500_000),      # discount_price 기준 (원)
    # UI 내부 상태
    "_page_visited": set(),
    "_data_health":  None,
}


def init_session() -> None:
    """페이지 진입 시 호출. 누락된 키만 채움 + 타입 정합화."""
    for k, v in _DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if hasattr(v, "copy") else v

    # brands는 항상 list[str] 이어야 멀티-pills와 호환 (이전 세션에서 단일 string 잔존 시 방어)
    if isinstance(st.session_state.get("brands"), str):
        st.session_state.brands = [st.session_state.brands]

    # year_range는 슬라이더 min/max 범위 내로 클램프
    yr = st.session_state.get("year_range")
    if yr is not None:
        lo, hi = tuple(yr)
        lo = max(2024, min(lo, 2026))
        hi = max(2024, min(hi, 2026))
        if (lo, hi) != tuple(yr):
            st.session_state.year_range = (lo, hi)


def get_filters() -> dict:
    """현재 필터 dict (얕은 복사). brands는 항상 list[str]로 정규화."""
    raw = st.session_state.brands
    brands = [raw] if isinstance(raw, str) else list(raw)
    return {
        "brands":       brands,
        "rating_sel":   st.session_state.rating_sel,
        "cat1_filters": list(st.session_state.cat1_filters),
        "cat2_filters": list(st.session_state.cat2_filters),
        "cat3_filters": list(st.session_state.cat3_filters),
        "year_range":   tuple(st.session_state.year_range),
        "price_range":  tuple(st.session_state.price_range),
    }


def reset_filters() -> None:
    """전체 필터 초기화 (체크박스 개별 키 포함)."""
    for k, v in _DEFAULTS.items():
        if not k.startswith("_"):
            st.session_state[k] = v.copy() if hasattr(v, "copy") else v
    # 개별 체크박스 키 초기화
    for k in list(st.session_state.keys()):
        if k.startswith("_cb_"):
            st.session_state[k] = False


def mark_page_visited(page_name: str) -> None:
    st.session_state._page_visited.add(page_name)


def filters_summary() -> str:
    """사이드바 하단 현재 필터 요약."""
    f = get_filters()
    parts = []
    n_brands = len(f["brands"])  # get_filters()가 항상 list 반환
    parts.append(f"브랜드 {n_brands}/4" if n_brands < 4 else "브랜드 전체")
    if f["rating_sel"] != "전체":
        parts.append(f"평점 {f['rating_sel']}")
    parts.append(f"연도 {f['year_range'][0]}~{f['year_range'][1]}")
    for key, label in [("cat1_filters", "카테고리1"), ("cat2_filters", "카테고리2"), ("cat3_filters", "카테고리3")]:
        if f[key]:
            parts.append(f"{label} {len(f[key])}개")
    lo, hi = f["price_range"]
    if lo > 0 or hi < 500_000:
        parts.append(f"가격 {lo:,}~{hi:,}원")
    return " · ".join(parts)
