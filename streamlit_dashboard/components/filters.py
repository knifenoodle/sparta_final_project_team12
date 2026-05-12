"""
filters.py — 사이드바 필터 위젯 (개선 버전)
================================

변경 사항:
  1. 순서 조정: 브랜드 > 평점 > 연도 > 가격 > 카테고리
  2. UI 업그레이드: 브랜드와 평점에 st.pills 적용 (더 직관적인 버튼 형태)
  3. 시각적 가독성 개선
"""
from __future__ import annotations

import streamlit as st
from config import BRAND_ORDER, BRANDS
from utils.session import filters_summary, reset_filters


def render_sidebar_filters(
    *,
    cat1_options: list[str] | None = None,
    cat2_options: list[str] | None = None,
    cat3_options: list[str] | None = None,
    price_min: int = 0,
    price_max: int = 500_000,
    lock_brand: str | None = None,
    show_year: bool = True,
    show_price: bool = True,
    show_category: bool = True,
) -> None:
    """사이드바 필터 렌더링. 분석 페이지 진입부에서 1회 호출.

    페이지별 활성 컬럼이 다른 경우 show_year/show_price/show_category로
    표시 항목을 조정한다 (BERTopic 페이지는 brand·rating·content만 보유).
    """

    sb = st.sidebar
    sb.markdown("## 필터 설정")
    sb.markdown("---")

    # ── 1. 브랜드 ──────────────────────────────────────────────
    sb.markdown("**브랜드**")
    if lock_brand:
        sb.info(f"{BRANDS[lock_brand]['label']}")
        st.session_state.brands = [lock_brand]
    else:
        # brands가 단일 string 잔존 시 list로 정합화 (multi-pills 호환)
        _b = st.session_state.get("brands")
        if isinstance(_b, str):
            st.session_state.brands = [_b]

        # 전체 선택/해제 버튼 (세션 스테이트 직접 조작)
        b_col1, b_col2 = sb.columns(2)
        if b_col1.button("전체 선택", use_container_width=True):
            st.session_state.brands = list(BRAND_ORDER)
            st.rerun()
        if b_col2.button("전체 해제", use_container_width=True):
            st.session_state.brands = []
            st.rerun()

        # 브랜드 선택 버튼형 UI
        sb.pills(
            "브랜드 선택",
            options=BRAND_ORDER,
            selection_mode="multi",
            key="brands",
            format_func=lambda b: BRANDS[b]["label"],
            label_visibility="collapsed" # 중복 라벨 숨김
        )

    sb.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── 2. 평점 ────────────────────────────────────────────────
    rating_options = ["전체", "1", "2", "3", "4", "5"]
    sb.pills(
        "평점",
        options=rating_options,
        selection_mode="single",
        key="rating_sel",
    )

    sb.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

    # ── 3. 연도 범위 ─────────────────────────────────────────────
    if show_year:
        yr = tuple(st.session_state.year_range)
        yr = (max(2024, min(yr[0], 2026)), max(2024, min(yr[1], 2026)))
        if yr != tuple(st.session_state.year_range):
            st.session_state.year_range = yr
        sb.slider(
            "연도 범위",
            min_value=2024,
            max_value=2026,
            value=tuple(st.session_state.year_range),
            step=1,
            key="year_range",
        )

    # ── 4. 가격 범위 ─────────────────────────────────────────────
    if show_price and price_max > price_min:
        stored_lo, stored_hi = st.session_state.price_range
        clamped_lo = max(price_min, min(stored_lo, price_max))
        clamped_hi = min(price_max, max(stored_hi, price_min))

        if (clamped_lo, clamped_hi) != (stored_lo, stored_hi):
            st.session_state.price_range = (clamped_lo, clamped_hi)

        sb.slider(
            "실구매가 범위 (원)",
            min_value=price_min,
            max_value=price_max,
            value=tuple(st.session_state.price_range),
            step=5_000,
            format="%d원",
            key="price_range",
        )

    # ── 5. 카테고리 필터 (기존 로직 유지) ──────────────────────────
    # 주의: `with sb.expander(...)` 안에서 위젯을 그릴 때는 `st.checkbox`(현재 컨텍스트)
    # 를 써야 하며, `sb.checkbox`는 사이드바 root에 그려져 expander 밖으로 새어 나간다.
    if show_category and (cat1_options or cat2_options or cat3_options):
        sb.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        sb.markdown("**카테고리 상세**")
        if cat1_options:
            with sb.expander("대분류", expanded=False):
                new_cat1 = [
                    opt for opt in cat1_options
                    if st.checkbox(opt, key=f"_cb_cat1_{opt}",
                                   value=(opt in st.session_state.cat1_filters))
                ]
                st.session_state.cat1_filters = new_cat1
        if cat2_options:
            with sb.expander("중분류", expanded=False):
                new_cat2 = [
                    opt for opt in cat2_options
                    if st.checkbox(opt, key=f"_cb_cat2_{opt}",
                                   value=(opt in st.session_state.cat2_filters))
                ]
                st.session_state.cat2_filters = new_cat2
        if cat3_options:
            with sb.expander("소분류", expanded=False):
                new_cat3 = [
                    opt for opt in cat3_options
                    if st.checkbox(opt, key=f"_cb_cat3_{opt}",
                                   value=(opt in st.session_state.cat3_filters))
                ]
                st.session_state.cat3_filters = new_cat3

    # ── 하단 요약 + 초기화 ────────────────────────────────────
    sb.markdown("---")
    sb.caption(f"**현재 설정 요약**")
    sb.caption(filters_summary())
    if sb.button("필터 초기화", use_container_width=True, type="primary"):
        reset_filters()
        st.rerun()