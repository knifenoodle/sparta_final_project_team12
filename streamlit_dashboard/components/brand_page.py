"""
brand_page.py — 브랜드 개별 페이지 공통 렌더러
================================================

P1(휠라) ~ P4(룰루레몬) 4개 페이지가 공유하는 단일 템플릿.
각 페이지 파일은 brand_key 하나만 넘기고 이 함수를 호출.

차트 구성:
  1. KPI 카드 (리뷰 수, 평균 평점)
  2. 연도별 매출액 추이 (외부 데이터 Placeholder) | 월별 리뷰 추이
  3. 카테고리별 리뷰 분포                        | 가격 분포 (discount_price)
  4. 리뷰 키워드 워드클라우드 (wordcloud 라이브러리 없으면 Top-30 bar)
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import BRANDS, BRAND_ORDER, BRAND_SALES
from utils.data_loader import get_reviews, get_tokens, apply_filters, filters_to_hash
from utils.session import init_session, get_filters, mark_page_visited
from utils.exceptions import safe_block, empty_state
from components.filters import render_sidebar_filters
from components.kpi_cards import metric_grid

try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    _HAS_WORDCLOUD = True
except ImportError:
    _HAS_WORDCLOUD = False

# 한국어 폰트 후보 (macOS / Linux)
_FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/Library/Fonts/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


# ─────────────────────────────────────────────────────────────
# 퍼블릭 진입점
# ─────────────────────────────────────────────────────────────
def render_brand_page(brand_key: str) -> None:
    """브랜드 페이지 공통 렌더러. 각 브랜드 페이지에서 1회 호출."""
    init_session()

    # ── 브랜드 선택 Pills ──────────────────────────────────────
    raw = st.session_state.get("brands", brand_key)
    current_brand = raw if isinstance(raw, str) else (raw[0] if raw else brand_key)

    selected = st.pills(
        "브랜드 선택",
        options=BRAND_ORDER,
        format_func=lambda b: BRANDS[b]["label"],
        default=current_brand,
        key="brand_pills",
    )
    effective_brand = selected if selected else current_brand
    if effective_brand != current_brand:
        st.session_state.brands = effective_brand
        st.rerun()

    mark_page_visited(f"brand_{effective_brand}")

    meta  = BRANDS[effective_brand]
    color = meta["color"]
    label = meta["label"]

    # ── 사이드바 필터 옵션 계산 (최소 컬럼만 로드) ────────────
    opts_df = get_reviews(columns=("brand", "cat1", "cat2", "cat3", "discount_price"))
    brand_opts = opts_df[opts_df["brand"] == effective_brand]

    cat1_opts = sorted(brand_opts["cat1"].dropna().unique().tolist())
    cat2_opts = sorted(brand_opts["cat2"].dropna().unique().tolist())
    cat3_opts = sorted(brand_opts["cat3"].dropna().unique().tolist())

    dp = brand_opts["discount_price"].dropna()
    p_min = int(dp.quantile(0.01)) if not dp.empty else 0
    p_max = int(dp.quantile(0.99)) if not dp.empty else 500_000

    render_sidebar_filters(
        cat1_options=cat1_opts,
        cat2_options=cat2_opts,
        cat3_options=cat3_opts,
        price_min=p_min,
        price_max=p_max,
        lock_brand=effective_brand,
    )

    # ── 필터 적용 ──────────────────────────────────────────────
    filters = get_filters()
    fh = filters_to_hash(filters)

    CHART_COLS = ("review_id", "brand", "cat1", "rating", "year", "month", "discount_price")
    reviews_all = get_reviews(columns=CHART_COLS)
    reviews_brand = reviews_all[reviews_all["brand"] == effective_brand]
    reviews_f = apply_filters(reviews_brand, filters)

    # ── 페이지 타이틀 ──────────────────────────────────────────
    st.markdown(
        f"<h2 style='border-left:5px solid {color}; padding-left:12px; margin-bottom:4px;'>"
        f"{label}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(f"필터 적용: {len(reviews_f):,} / 전체 {len(reviews_brand):,}건")

    if reviews_f.empty:
        empty_state("필터 결과 0건", "사이드바에서 필터 조건을 넓혀 주세요.")
        return

    # ── KPI ────────────────────────────────────────────────────
    n_reviews  = len(reviews_f)
    avg_rating = float(reviews_f["rating"].mean()) if "rating" in reviews_f.columns else 0.0

    metric_grid([
        {"label": "리뷰 수",    "value": f"{n_reviews:,}",    "help": "필터 적용 기준"},
        {"label": "평균 평점",  "value": f"{avg_rating:.2f}", "help": "1~5점 기준"},
    ], cols=2)

    st.divider()

    # ── 행 1: 매출 추이 Placeholder + 월별 리뷰 추이 ──────────
    col_a, col_b = st.columns(2)
    with col_a:
        with safe_block("매출 추이"):
            # 매출 차트는 pills로 선택된 effective_brand 기준으로 그려야 함
            # (이전: 호출 시 인자 brand_key로 고정 → FILA 매출만 표시되는 버그)
            st.plotly_chart(
                _sales_placeholder(effective_brand, color),
                use_container_width=True,
            )
    with col_b:
        with safe_block("월별 리뷰 추이"):
            st.plotly_chart(
                _monthly_trend(reviews_f, color, label),
                use_container_width=True,
            )

    st.divider()

    # ── 행 2: 카테고리 분포 + 가격 분포 ──────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        with safe_block("카테고리 분포"):
            st.plotly_chart(
                _category_bar(reviews_f, color, label),
                use_container_width=True,
            )
    with col_d:
        with safe_block("가격 분포"):
            st.plotly_chart(
                _price_histogram(reviews_f, color, label),
                use_container_width=True,
            )

    st.divider()

    # ── 워드클라우드 (full width) ──────────────────────────────
    st.subheader("리뷰 주요 키워드")
    st.caption(
        "preprocessed_absa.parquet의 `tokens_topic` 컬럼 사용 — "
        "Kiwi 형태소 분석(사용자/불용/정규화 사전) 적용된 어휘"
    )
    with safe_block("워드클라우드"):
        token_df = get_tokens(brand=effective_brand, column="tokens_topic")
        brand_tokens = token_df["tokens_topic"] if "tokens_topic" in token_df.columns else pd.Series(dtype="object")
        _render_wordcloud(brand_tokens, color, label)

    # ── 긍정/부정/카테고리 Top 키워드 (3패널) ────────────────
    st.markdown(
        "<div style='height:8px'></div><h5 style='margin:0;'>긍정 · 부정 · 카테고리별 Top 키워드</h5>",
        unsafe_allow_html=True,
    )
    st.caption(
        "별점 기준 분리 — 긍정(4–5점) / 부정(1–2점) / 카테고리(cat1)별 Top1 카테고리 키워드. "
        "워드클라우드보다 전략 해석이 명확함"
    )
    with safe_block("Top 키워드 3패널"):
        _render_keyword_trio(effective_brand, color, label)


# ─────────────────────────────────────────────────────────────
# 내부 차트 헬퍼
# ─────────────────────────────────────────────────────────────
def _sales_placeholder(brand_key: str, color: str) -> go.Figure:
    s = BRAND_SALES.get(brand_key, {})
    years = [2023, 2024, 2025]
    vals  = [s.get("2023", 0), s.get("2024", 0), s.get("2025", 0)]
    x_labels = [str(y) for y in years]
    if s.get("est") and x_labels:
        x_labels[-1] = "2025*"

    fig = go.Figure()
    fig.add_bar(
        x=x_labels, y=vals,
        marker_color=color,
        hovertemplate="%{x}년: %{y:,}억 원<extra></extra>",
    )
    if s.get("est"):
        fig.add_annotation(
            text="* 2025 추산치 (회계연도 기준 성장흐름 유지 가정)",
            xref="paper", yref="paper", x=0.5, y=1.07,
            showarrow=False, font=dict(size=11, color="#888"),
        )
    fig.update_layout(
        title="연도별 매출액 추이 (억 원)",
        height=340,
        xaxis=dict(tickvals=x_labels),
        yaxis_title="매출액 (억 원)",
        showlegend=False,
    )
    return fig


def _monthly_trend(df: pd.DataFrame, color: str, label: str) -> go.Figure:
    if df.empty or "year" not in df.columns or "month" not in df.columns:
        return _empty_fig("월별 추이 데이터 없음")
    agg = (
        df.groupby(["year", "month"])
        .size()
        .reset_index(name="n")
    )
    agg["period"] = agg["year"].astype(str) + "-" + agg["month"].apply(lambda m: f"{m:02d}")
    agg = agg.sort_values("period")
    fig = px.line(
        agg, x="period", y="n",
        markers=True,
        labels={"period": "연-월", "n": "리뷰 수"},
        title="월별 리뷰 추이",
    )
    fig.update_traces(line_color=color, marker_color=color)

    # 변곡점 자동 주석 — 최댓값 지점에 annotation
    if not agg.empty:
        peak = agg.loc[agg["n"].idxmax()]
        fig.add_annotation(
            x=peak["period"], y=peak["n"],
            text=f"최댓값 {peak['n']:,}건",
            showarrow=True, arrowhead=2, arrowsize=1.0,
            ax=0, ay=-30,
            font=dict(size=10, color=color),
            bgcolor="rgba(255,255,255,0.85)",
            borderpad=2,
        )
    # 데이터 수집 기준 안내
    fig.add_annotation(
        text="ℹ 동일 수집 기준 — 변곡점은 수요 변화/이벤트 가능성",
        xref="paper", yref="paper", x=0.99, y=1.10,
        showarrow=False, font=dict(size=10, color="#888"),
        xanchor="right",
    )
    fig.update_layout(height=360, hovermode="x unified", margin=dict(t=70))
    return fig


def _category_bar(df: pd.DataFrame, color: str, label: str) -> go.Figure:
    if df.empty or "cat1" not in df.columns:
        return _empty_fig("카테고리 데이터 없음")
    cat = df["cat1"].value_counts().head(10).reset_index()
    cat.columns = ["카테고리", "리뷰 수"]
    fig = px.bar(
        cat, x="리뷰 수", y="카테고리", orientation="h",
        labels={"리뷰 수": "리뷰 수", "카테고리": ""},
        title="카테고리별 리뷰 분포 (Top 10)",
        color_discrete_sequence=[color],
    )
    fig.update_layout(height=340, yaxis=dict(autorange="reversed"))
    return fig


def _price_histogram(df: pd.DataFrame, color: str, label: str) -> go.Figure:
    if df.empty or "discount_price" not in df.columns:
        return _empty_fig("가격 데이터 없음")
        
    prices = df["discount_price"].dropna()
    prices = prices[prices > 0]
    
    if prices.empty:
        return _empty_fig("가격 > 0 데이터 없음")
        
    cap = prices.quantile(0.97)
    prices = prices[prices <= cap]
    
    # 1. 일단 기본 차트를 그립니다.
    fig = px.histogram(
        prices, 
        x=prices,
        nbins=40,
        color_discrete_sequence=[color],
    )
    
    # 💡 2. 확실한 강제 덮어쓰기: 제목, Y축, 레이아웃 일괄 업데이트
    fig.update_layout(
        title="상품 실구매가 분포 (상위 3% 이상치 제외)", # 제목 완벽 한글화
        yaxis_title="상품 수 (건)",                   # Y축 한글화
        height=340, 
        bargap=0.05
    )
    
    # 💡 3. X축 이름 강제 덮어쓰기 (discount_price를 무시하고 적용됨)
    fig.update_xaxes(title_text="실구매가 (원)")
    
    return fig


# 워드클라우드 추가 불용어 — 형태소 분석 후에도 남는 일반 평가 어휘 완벽 필터링
_EXTRA_STOPWORDS = {
    # 1. 문법적 뼈대 및 의미 없는 동사/형용사
    "하다", "있다", "없다", "같다", "이다", "되다", "나다", "그렇다", "가다", "오다", "다니다", "들다", "받다", "사다", "주다", "맞추다",
    "것", "수", "때", "거", "게", "걸", "더", "안", "못",

    # 2. 강도 부사 및 일반 부사 (워드클라우드에서 덩치만 키우는 주범)
    "너무", "진짜", "정말", "그냥", "조금", "약간", "엄청", "완전", "되게", "매우", "많이", "아주", "역시", "딱", "제일", "가장", "자주", "항상",

    # 3. 리뷰/플랫폼 관련 기본 단어 및 일반 명사
    "구매", "주문", "배송", "후기", "리뷰", "네이버페이", "작성", "제품", "상품", "생각", "마음", "느낌", "부분", "정도", "처음", "이번", "하나", 

    # 4. 뻔한 '긍정/평가' 어휘 (현재 워드클라우드를 집어삼킨 거인들)
    "좋다", "예쁘다", "편하다", "잘맞다", "어울리다", "마음에들다", "만족", "최고", "이쁘다", "가볍다", "부드럽다", "괜찮다", "귀엽다", "추천", "감사",

    # 5. 1차원적 카테고리 명사 및 상태 (이걸 지워야 진짜 '특징'이 보입니다)
    "사이즈", "신발", "운동화", "디자인", "색상", "컬러", "색감", "정사이즈", "착용감", "착용",
    "크다", "작다", "맞다", "입다", "신다", "길다", "짧다",

    # 6. 브랜드명
    "휠라", "룰루레몬", "젝시믹스", "안다르"
}

# (참고) 혹시 토큰화 방식에 따라 활용형이 남아있을 수 있으므로 기존 파생형 일부 유지
_EXTRA_STOPWORDS.update({
    "좋은", "좋아", "좋네", "좋고", "좋았", "좋아요", "괜찮아", 
    "있는", "있어", "있고", "없는", "없어", "같아", "같은", "같이", "되는", "된다",
    "사서", "받았", "받은", "그런", "이런", "저런", "어떤",
    "하는", "한", "해서", "해요", "합니다", "했어요", "했다",
    "그리고", "근데", "하지만"
})


def _render_wordcloud(token_series: pd.Series, color: str, label: str) -> None:
    """워드클라우드 또는 Top-30 bar 폴백."""
    tokens_all = " ".join(token_series.dropna().astype(str)).split()
    freq = Counter(tokens_all)
    # 불용어 제거: 길이 1 / 숫자만 / 일반 평가 어휘
    freq = {
        w: c for w, c in freq.items()
        if len(w) > 1 and not w.isdigit() and w not in _EXTRA_STOPWORDS
    }

    if not freq:
        empty_state("토큰 데이터 없음")
        return

    if _HAS_WORDCLOUD:
        font_path = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
        wc = WordCloud(
            width=1000, height=400,
            background_color="#0e1117",
            colormap="Blues",
            font_path=font_path,
            max_words=100,
            prefer_horizontal=0.85,
        )
        wc.generate_from_frequencies(freq)
        fig_wc, ax = plt.subplots(figsize=(14, 5))
        fig_wc.patch.set_facecolor("#0e1117")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wc, use_container_width=True)
        plt.close(fig_wc)
    else:
        # 폴백: Top-30 bar chart
        top30 = pd.DataFrame(
            sorted(freq.items(), key=lambda x: x[1], reverse=True)[:30],
            columns=["단어", "빈도"],
        )
        fig = px.bar(
            top30.sort_values("빈도"), x="빈도", y="단어", orientation="h",
            title="주요 키워드 Top 30 (wordcloud 미설치 — bar 대체)",
            color_discrete_sequence=[color],
        )
        fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("`uv add wordcloud matplotlib` 후 재실행 시 워드클라우드로 전환됩니다.")


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="gray"),
    )
    fig.update_layout(height=320, xaxis_visible=False, yaxis_visible=False)
    return fig


def _tokens_to_counter(token_series: pd.Series) -> Counter:
    """tokens_topic Series → 빈도 Counter (불용어·1글자·숫자 제외)."""
    out: Counter = Counter()
    for val in token_series.dropna().astype(str):
        for w in val.split():
            if len(w) > 1 and not w.isdigit() and w not in _EXTRA_STOPWORDS:
                out[w] += 1
    return out


def _topn_bar(freq: Counter, color: str, title: str, n: int = 10) -> go.Figure:
    """Counter → horizontal bar (Top n)."""
    if not freq:
        return _empty_fig(f"{title} 데이터 없음")
    top = pd.DataFrame(
        sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n],
        columns=["단어", "빈도"],
    )
    fig = px.bar(
        top.sort_values("빈도"), x="빈도", y="단어",
        orientation="h",
        color_discrete_sequence=[color],
        title=title,
        labels={"빈도": "빈도", "단어": ""},
    )
    fig.update_layout(
        height=320,
        margin=dict(t=40, b=20, l=10, r=10),
        showlegend=False,
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    return fig


def _render_keyword_trio(brand: str, color: str, label: str) -> None:
    """긍정 / 부정 / 카테고리 Top1 의 Top 10 키워드 막대 3패널."""
    df = get_reviews(columns=("brand", "rating", "cat1", "tokens_topic"))
    df = df[df["brand"] == brand]
    if df.empty:
        empty_state("키워드 분석용 데이터 없음")
        return

    pos = df[df["rating"] >= 4]["tokens_topic"]
    neg = df[df["rating"] <= 2]["tokens_topic"]

    top_cat = df["cat1"].value_counts().head(1)
    top_cat_name = top_cat.index[0] if not top_cat.empty else None
    cat_tokens = df[df["cat1"] == top_cat_name]["tokens_topic"] if top_cat_name else pd.Series(dtype="object")

    freq_pos = _tokens_to_counter(pos)
    freq_neg = _tokens_to_counter(neg)
    freq_cat = _tokens_to_counter(cat_tokens)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(
            _topn_bar(freq_pos, "#1565C0", f"긍정(4–5점) Top 10 — {len(pos):,}건"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            _topn_bar(freq_neg, "#C62828", f"부정(1–2점) Top 10 — {len(neg):,}건"),
            use_container_width=True,
        )
    with c3:
        cat_title = (
            f"카테고리 ‘{top_cat_name}’ Top 10 — {len(cat_tokens):,}건"
            if top_cat_name else "카테고리 데이터 없음"
        )
        st.plotly_chart(
            _topn_bar(freq_cat, color, cat_title),
            use_container_width=True,
        )

    st.caption(
        f"긍정·부정은 별점 임계 기준(4↑/2↓), 카테고리는 {label}의 cat1 분포 1위 카테고리 기준. "
        "불용어·1글자·숫자 제외. 더 깊은 속성 단위(P/N) 분리는 ABSA 페이지 참조"
    )
