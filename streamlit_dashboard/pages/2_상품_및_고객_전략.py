"""
2_상품_및_고객_전략.py — 상품 및 고객 전략 분석
================================================

정형 데이터 기반 비즈니스 분석 (preprocessed_absa.parquet 실데이터):
  1. 체형별 만족도        — 키·몸무게 구간별 평균 평점 Heatmap
  2. 할인율별 평점 분포   — 할인율(%) vs 평균 평점 Scatter (프로모션 강도 vs 만족도)
  3. 고관여 인게이지먼트  — 포토 리뷰 비율 + 평균 도움이 돼요 + 신상품 포토 비율
  4. SKU 복잡도           — 색상 수 구간별 리뷰 수 분포 Box Plot
  5. 컬러 빈도 분석       — 브랜드별 구매 옵션 상위 컬러
  6. 리뷰 볼륨 x 평점     — 브랜드 x 카테고리 사분면 Scatter
  7. Action Recommendation — 상품 전략 요약 카드
"""
from __future__ import annotations

import sys
from pathlib import Path

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import APP_TITLE, PAGE_LAYOUT, BRANDS, BRAND_ORDER, CACHE_TTL
from utils.data_loader import get_reviews
from utils.session import init_session, get_filters, mark_page_visited
from utils.exceptions import safe_block, empty_state
from components.filters import render_sidebar_filters
from components.page_header import render_page_intro

st.set_page_config(
    page_title=f"{APP_TITLE} — 상품·고객 전략",
    page_icon=None,
    **PAGE_LAYOUT,
)
init_session()
mark_page_visited("product_strategy")

st.title("상품 및 고객 전략")
st.caption("체형 적합성 · 할인율별 평점 · 고관여 인게이지먼트 · SKU 복잡도 · 컬러 빈도 · 사분면 분석")
st.markdown(
    "<p style='font-size:12px; color:#888; margin:2px 0 2px;'>"
    "홈 &nbsp;›&nbsp; <strong>상품/고객 전략</strong> &nbsp;›&nbsp; "
    "BERTopic &nbsp;›&nbsp; ABSA &nbsp;›&nbsp; 포지셔닝</p>",
    unsafe_allow_html=True,
)
st.caption("홈의 브랜드별 KPI에서 파악한 점유율·평점 차이를 정형 데이터(체형·가격·인게이지먼트)로 심화 분석합니다.")

render_page_intro(
    "카테고리·가격·체형·할인율·인게이지먼트 단위로 침투 가능 영역을 도출하여 "
    "FILA 의류 시장 진입의 1차 타겟 세그먼트를 좁힙니다.",
    accent="#003087",
)

render_sidebar_filters()
filters = get_filters()
active_brands = [b for b in BRAND_ORDER if b in filters.get("brands", BRAND_ORDER)]
if not active_brands:
    active_brands = BRAND_ORDER

_CMAP = {b: BRANDS[b]["color"] for b in BRAND_ORDER}

# 한국어 컬러명 → Hex 매핑 (라이트 모드 가독성 기준)
_KR_COLOR_HEX: dict[str, str] = {
    "블랙": "#1A1A1A", "검정": "#1A1A1A", "블랙계열": "#1A1A1A",
    "화이트": "#AAAAAA", "흰색": "#AAAAAA", "흰": "#AAAAAA",
    "네이비": "#003087", "남색": "#003087", "네이비계열": "#003087",
    "그레이": "#757575", "회색": "#757575", "그레이계열": "#757575",
    "베이지": "#C8AA6E", "베이지계열": "#C8AA6E",
    "아이보리": "#C0B080", "크림": "#C0B080",
    "핑크": "#E91E8C", "분홍": "#F06292", "핑크계열": "#E91E8C",
    "레드": "#CC0000", "빨강": "#CC0000", "레드계열": "#CC0000",
    "블루": "#1565C0", "파랑": "#1565C0", "블루계열": "#1565C0",
    "그린": "#2E7D32", "초록": "#2E7D32", "그린계열": "#2E7D32",
    "민트": "#00897B", "민트계열": "#00897B",
    "카키": "#827717", "올리브": "#556B2F",
    "브라운": "#5D4037", "갈색": "#795548", "카멜": "#795548", "브라운계열": "#5D4037",
    "퍼플": "#6A1B9A", "보라": "#7B1FA2", "자주": "#880E4F",
    "옐로우": "#F9A825", "노랑": "#F57F17", "옐로우계열": "#F9A825",
    "오렌지": "#E65100", "오렌지계열": "#E65100",
    "라임": "#558B2F", "연두": "#7CB342",
    "코랄": "#E64A4A",
    "차콜": "#37474F", "챠콜": "#37474F", "차콜계열": "#37474F",
    "스카이블루": "#0288D1", "하늘": "#0288D1",
    "와인": "#880E4F", "버건디": "#880E4F",
    "머스타드": "#F57F17",
    "샌드": "#A1887F",
    "딥블루": "#0D47A1",
    "연분홍": "#AD1457",
    "골드": "#C79100",
    "실버": "#757575",
    "다크그린": "#1B5E20",
    "라이트블루": "#0277BD",
}

# ── 실데이터 집계 함수 ───────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL)
def _body_heatmap_df(brand: str) -> pd.DataFrame:
    df = get_reviews(columns=("brand", "user_height_group", "user_weight_group", "rating"))
    df = df[
        (df["brand"] == brand) &
        (df["user_height_group"] != "unknown") &
        (df["user_weight_group"] != "unknown")
    ]
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(["user_height_group", "user_weight_group"])["rating"].agg(
        평균_평점="mean", 리뷰수="count"
    ).reset_index()
    agg.columns = ["키 구간", "몸무게 구간", "평균 평점", "리뷰 수"]
    agg["평균 평점"] = agg["평균 평점"].round(2)
    return agg


@st.cache_data(ttl=CACHE_TTL)
def _price_elasticity_df(brand: str) -> pd.DataFrame:
    df = get_reviews(columns=("brand", "original_price", "discount_price", "rating"))
    df = df[
        (df["brand"] == brand) &
        (df["original_price"] > 0) &
        (df["discount_price"] > 0)
    ].copy()
    if df.empty:
        return pd.DataFrame()
    df["할인율 (%)"] = (
        (df["original_price"] - df["discount_price"]) / df["original_price"] * 100
    ).clip(0, 100)
    # 5% 구간 집계 — scatter 가독성 및 추세선 안정성 확보
    df["구간"] = (df["할인율 (%)"] // 5 * 5).round(0)
    agg = df.groupby("구간").agg(
        평균_평점=("rating", "mean"),
        리뷰_수=("rating", "count"),
    ).reset_index()
    agg.columns = ["할인율 (%)", "평균 평점", "리뷰 수"]
    agg["평균 평점"] = agg["평균 평점"].round(2)
    return agg


@st.cache_data(ttl=CACHE_TTL)
def _engagement_df() -> pd.DataFrame:
    df = get_reviews(columns=("brand", "has_image", "helpful_count", "is_new", "review_date", "collect_date"))
    df = df.copy()
    df["is_new_bool"] = df["is_new"].isin(["True", "1.0"]).astype(int)

    # 노출 기간 보정용 일수 계산 (수집일 - 작성일, 최소 1일)
    if "review_date" in df.columns and "collect_date" in df.columns:
        df["review_date"]  = pd.to_datetime(df["review_date"], errors="coerce")
        df["collect_date"] = pd.to_datetime(df["collect_date"], errors="coerce")
        df["days_exposed"] = (df["collect_date"] - df["review_date"]).dt.days.clip(lower=1)
    else:
        df["days_exposed"] = 1

    rows = []
    for brand in BRAND_ORDER:
        sub = df[df["brand"] == brand]
        if sub.empty:
            continue
        photo_ratio = float(sub["has_image"].mean())
        helpful_avg = float(sub["helpful_count"].mean())
        # 노출 기간 정규화: helpful_count / days_exposed (일평균 도움이 돼요)
        helpful_per_day = float((sub["helpful_count"] / sub["days_exposed"]).mean())
        new_sub = sub[sub["is_new_bool"] == 1]
        new_photo = float(new_sub["has_image"].mean()) if len(new_sub) > 0 else 0.0
        rows.append({
            "브랜드":           BRANDS[brand]["label"],
            "brand_key":        brand,
            "포토 리뷰 비율":     round(photo_ratio, 4),
            "평균 도움이 돼요":    round(helpful_avg, 2),
            "일평균 도움이 돼요":  round(helpful_per_day, 4),
            "신상품 포토 비율":   round(new_photo, 4),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL)
def _sku_review_counts_df(brand: str) -> pd.DataFrame:
    df = get_reviews(columns=("review_id", "brand", "product_id", "color_count"))
    df = df[(df["brand"] == brand) & (df["color_count"] > 0)]
    if df.empty:
        return pd.DataFrame()
    prod = df.groupby("product_id").agg(
        리뷰_수=("review_id", "count"),
        color_count=("color_count", "first"),
    ).reset_index()

    def _bucket(c: float) -> str:
        if c <= 1:  return "1가지"
        if c <= 3:  return "2~3가지"
        if c <= 6:  return "4~6가지"
        if c <= 10: return "7~10가지"
        return "11가지 이상"

    prod["색상 수 구간"] = prod["color_count"].apply(_bucket)
    return prod[["색상 수 구간", "리뷰_수"]].rename(columns={"리뷰_수": "리뷰 수"})


@st.cache_data(ttl=CACHE_TTL)
def _color_freq_df(brand: str) -> pd.DataFrame:
    df = get_reviews(columns=("brand", "purchase_option_color"))
    df = df[(df["brand"] == brand) & (df["purchase_option_color"] != "unknown")]
    if df.empty:
        return pd.DataFrame()
    # 복합색(쉼표 구분) → 첫 번째 색상만 사용
    colors = df["purchase_option_color"].str.split(",").str[0].str.strip()
    counts = colors.value_counts().head(10).reset_index()
    counts.columns = ["컬러", "언급 수"]
    return counts


@st.cache_data(ttl=CACHE_TTL)
def _quadrant_df() -> pd.DataFrame:
    df = get_reviews(columns=("review_id", "brand", "cat1", "rating"))
    df = df[df["cat1"].notna() & (df["cat1"] != "")]
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(["brand", "cat1"]).agg(
        리뷰수=("review_id", "count"),
        평균평점=("rating", "mean"),
    ).reset_index()
    agg.columns = ["brand_key", "카테고리", "리뷰 수", "평균 평점"]
    agg["브랜드"] = agg["brand_key"].map(lambda b: BRANDS[b]["label"])
    agg["평균 평점"] = agg["평균 평점"].round(2)
    return agg


# ─────────────────────────────────────────────────────────────
# 1. 체형별 만족도 Heatmap
# ─────────────────────────────────────────────────────────────
st.subheader("체형별 만족도")
st.caption("키·몸무게 구간별 평균 평점 — 사이즈 사각지대(White Space) 및 핵심 체형 타겟 탐색")

with safe_block("체형 Heatmap"):
    # 사이드바 선택 브랜드 기반 탭 생성
    tabs = st.tabs([BRANDS[b]["label"] for b in active_brands])
    
    for tab, brand in zip(tabs, active_brands):
        with tab:
            body_df = _body_heatmap_df(brand)
            if body_df.empty:
                empty_state("체형 데이터 없음", "user_height_group / user_weight_group 값 확인 필요")
            else:
                pivot = body_df.pivot_table(
                    index="몸무게 구간", columns="키 구간",
                    values="평균 평점", aggfunc="mean",
                )
                
                def _first_num(s: str) -> int:
                    m = re.search(r"\d+", s)
                    return int(m.group()) if m else 0

                height_order = sorted(pivot.columns.tolist(), key=_first_num)
                weight_order = sorted(pivot.index.tolist(), key=_first_num)
                pivot = pivot.reindex(index=weight_order, columns=height_order)
                
                fig_body = px.imshow(
                    pivot,
                    color_continuous_scale="RdBu",
                    zmin=1, zmax=5,
                    aspect="auto",
                    text_auto=".2f",
                    labels=dict(color="평균 평점", x="키 구간", y="몸무게 구간"),
                    title=f"{BRANDS[brand]['label']} — 체형별 평균 평점",
                )
                fig_body.update_layout(height=420)
                st.plotly_chart(fig_body, use_container_width=True)
                st.caption(f"집계 기준: 체형 정보가 포함된 리뷰 {body_df['리뷰 수'].sum():,}건 (unknown 제외)")

st.divider()

# ─────────────────────────────────────────────────────────────
# 2. 할인율별 평점 분포 (구 "가격 탄력성" — 명칭 정정)
# ─────────────────────────────────────────────────────────────
st.subheader("할인율별 평점 분포")
st.caption(
    "할인율(%) 구간별 평균 평점 — 프로모션 강도와 만족도의 관계 관찰. "
    "(엄밀한 가격 탄력성은 가격 변화 → 수요량(판매량/전환율) 변화로 정의되며, "
    "본 차트는 만족도 측면의 보조 지표로 해석)"
)

with safe_block("할인율 × 평점 Scatter"):
    pe_cols = st.columns(len(active_brands))
    for col, brand in zip(pe_cols, active_brands):
        with col:
            pe_df = _price_elasticity_df(brand)
            if pe_df.empty:
                empty_state(f"{BRANDS[brand]['label']} 데이터 없음")
            else:
                fig_pe = px.scatter(
                    pe_df,
                    x="할인율 (%)", y="평균 평점",
                    size="리뷰 수",
                    color_discrete_sequence=[BRANDS[brand]["color"]],
                    opacity=0.70,
                    trendline="lowess",
                    labels={"할인율 (%)": "할인율 (%)", "평균 평점": "평점"},
                    title=BRANDS[brand]["label"],
                )
                fig_pe.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig_pe, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 3. 고관여 고객 인게이지먼트
# ─────────────────────────────────────────────────────────────
st.subheader("고관여 고객 인게이지먼트")
st.caption(
    "포토 리뷰 비율(has_image) · 평균 도움이 돼요(helpful_count) · 신상품 출시 포토 비율(is_new) "
    "— 브랜드별 팬덤 화력 및 UGC 질 비교"
)

with safe_block("인게이지먼트"):
    eng_df = _engagement_df()
    if eng_df.empty:
        empty_state("인게이지먼트 데이터 없음")
    else:
        normalize_helpful = st.toggle(
            "노출 기간 보정 (일평균)",
            value=False,
            help="helpful_count는 리뷰 노출 기간이 길수록 누적되므로 "
                 "일수로 나눠 정규화한 일평균 값을 표시합니다.",
            key="helpful_normalize",
        )
        e1, e2, e3 = st.columns(3)

        with e1:
            fig_e1 = px.bar(
                eng_df,
                x="브랜드", y="포토 리뷰 비율",
                color="brand_key",
                color_discrete_map=_CMAP,
                title="포토 리뷰 비율 (has_image)",
                labels={"포토 리뷰 비율": "비율"},
                text=eng_df["포토 리뷰 비율"].apply(lambda x: f"{x:.1%}"),
            )
            y_max = max(eng_df["포토 리뷰 비율"].max() * 1.2, 0.1)
            fig_e1.update_layout(height=360, showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, y_max])
            fig_e1.update_traces(textposition="outside")
            st.plotly_chart(fig_e1, use_container_width=True)

        with e2:
            metric_col = "일평균 도움이 돼요" if normalize_helpful else "평균 도움이 돼요"
            metric_title = (
                "일평균 도움이 돼요 (helpful_count / days_exposed)"
                if normalize_helpful else
                "평균 도움이 돼요 (helpful_count)"
            )
            text_fmt = "{:.4f}" if normalize_helpful else "{:.2f}"

            fig_e2 = px.bar(
                eng_df,
                x="브랜드", y=metric_col,
                color="brand_key",
                color_discrete_map=_CMAP,
                title=metric_title,
                labels={metric_col: "평균 수"},
                text=eng_df[metric_col].apply(text_fmt.format),
            )
            y_max2 = max(eng_df[metric_col].max() * 1.3, 0.5 if not normalize_helpful else 0.01)
            fig_e2.update_layout(height=360, showlegend=False, yaxis_range=[0, y_max2])
            fig_e2.update_traces(textposition="outside")
            st.plotly_chart(fig_e2, use_container_width=True)

        with e3:
            fig_e3 = px.bar(
                eng_df,
                x="브랜드", y="신상품 포토 비율",
                color="brand_key",
                color_discrete_map=_CMAP,
                title="신상품 출시 포토 비율 (is_new=True)",
                labels={"신상품 포토 비율": "비율"},
                text=eng_df["신상품 포토 비율"].apply(lambda x: f"{x:.1%}"),
            )
            y_max3 = max(eng_df["신상품 포토 비율"].max() * 1.2, 0.1)
            fig_e3.update_layout(height=360, showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, y_max3])
            fig_e3.update_traces(textposition="outside")
            st.plotly_chart(fig_e3, use_container_width=True)

        with st.expander("고관여 원문 탐색 — 튀는 값 리뷰 확인"):
            oc1, oc2 = st.columns(2)
            with oc1:
                out_brand = st.selectbox(
                    "브랜드",
                    options=active_brands,
                    format_func=lambda b: BRANDS[b]["label"],
                    key="outlier_brand",
                )
            with oc2:
                out_metric = st.radio(
                    "이상치 기준 지표",
                    ["helpful_count (도움이 돼요)", "has_image (포토 리뷰)"],
                    horizontal=True,
                    key="outlier_metric",
                )
            metric_col = "helpful_count" if "helpful_count" in out_metric else "has_image"

            if metric_col == "helpful_count":
                threshold = st.slider("최솟값 (이 값 이상만 표시)", 1, 100, 5, key="outlier_threshold")
            else:
                threshold = 1

            raw_df = get_reviews(
                columns=("review_id", "brand", "rating", "helpful_count", "has_image", "content_clean")
            )
            sub_raw = raw_df[raw_df["brand"] == out_brand].copy()
            if metric_col == "helpful_count":
                sub_raw = sub_raw[sub_raw["helpful_count"] >= threshold]
            else:
                sub_raw = sub_raw[sub_raw["has_image"] == 1]
            sub_raw = sub_raw.sort_values(metric_col, ascending=False).head(50)

            if sub_raw.empty:
                empty_state("해당 조건의 리뷰 없음", "기준값을 낮추거나 브랜드를 변경해 주세요.")
            else:
                st.caption(f"{len(sub_raw):,}건 표시 (최대 50건, {metric_col} 내림차순)")
                st.dataframe(
                    sub_raw[["review_id", "rating", "helpful_count", "has_image", "content_clean"]],
                    use_container_width=True,
                    hide_index=True,
                )

st.divider()

# ─────────────────────────────────────────────────────────────
# 4. SKU 복잡도 — 색상 수 구간 x 리뷰 수 Box Plot
# ─────────────────────────────────────────────────────────────
st.subheader("SKU 복잡도 분석")
st.caption("색상 수 구간별 리뷰 수 분포 — 컬러 라인업 확장이 판매 성과에 미치는 영향")

with safe_block("SKU Box Plot"):
    sku_cols = st.columns(len(active_brands))
    _sku_order = ["1가지", "2~3가지", "4~6가지", "7~10가지", "11가지 이상"]
    for col, brand in zip(sku_cols, active_brands):
        with col:
            sku_df = _sku_review_counts_df(brand)
            if sku_df.empty:
                empty_state(f"{BRANDS[brand]['label']} 데이터 없음")
            else:
                fig_sku = px.box(
                    sku_df,
                    x="색상 수 구간",
                    y="리뷰 수",
                    color_discrete_sequence=[BRANDS[brand]["color"]],
                    title=BRANDS[brand]["label"],
                    labels={"색상 수 구간": "컬러 수", "리뷰 수": "리뷰 수"},
                    category_orders={"색상 수 구간": _sku_order},
                )
                fig_sku.update_layout(height=340, showlegend=False)
                st.plotly_chart(fig_sku, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 5. 컬러 빈도 분석
# ─────────────────────────────────────────────────────────────
st.subheader("컬러 빈도 분석")
st.caption(
    "구매 옵션 컬러(purchase_option_color) 언급 빈도 Top 10 "
    "— 브랜드별 주력 컬러 팔레트 파악 및 휠라 초기 컬러 전략 수립"
)

def _resolve_color_hex(name: str) -> str:
    """정확 매칭 → 부분 문자열 매칭(베이스 컬러) → 기본 회색 순으로 hex 반환."""
    if name in _KR_COLOR_HEX:
        return _KR_COLOR_HEX[name]
    for key in sorted(_KR_COLOR_HEX, key=len, reverse=True):
        if key in name:
            return _KR_COLOR_HEX[key]
    return "#BBBBBB"


with safe_block("컬러 Bar"):
    cf_cols = st.columns(len(active_brands))
    for col, brand in zip(cf_cols, active_brands):
        with col:
            cf_df = _color_freq_df(brand)
            if cf_df.empty:
                empty_state(f"{BRANDS[brand]['label']} 데이터 없음")
            else:
                sorted_df = cf_df.sort_values("언급 수")
                y_labels = [
                    f"<span style='color:{_resolve_color_hex(c)}'>●</span> {c}"
                    for c in sorted_df["컬러"]
                ]
                # 막대는 브랜드 단색으로 통일 (가독성 우선)
                fig_cf = go.Figure(go.Bar(
                    x=sorted_df["언급 수"],
                    y=y_labels,
                    orientation="h",
                    marker_color=BRANDS[brand]["color"],
                    marker_line_width=0,
                    text=sorted_df["언급 수"].apply(lambda v: f"{v:,}"),
                    textposition="outside",
                    hovertemplate="%{y}: %{x:,}건<extra></extra>",
                ))
                fig_cf.update_layout(
                    title=BRANDS[brand]["label"],
                    height=380,
                    showlegend=False,
                    xaxis_title="언급 수",
                    yaxis=dict(tickfont=dict(size=12)),
                )
                st.plotly_chart(fig_cf, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 6. 리뷰 볼륨 x 평점 사분면
# ─────────────────────────────────────────────────────────────
st.subheader("리뷰 볼륨 x 평점 사분면")
st.caption(
    "브랜드 x 카테고리별 리뷰 수(볼륨) vs 평균 평점(만족도) "
    "— 고성장·고만족 카테고리 탐색 및 휠라 진입 기회 영역 식별"
)

with safe_block("사분면 Scatter"):
    quad_df = _quadrant_df()
    if quad_df.empty:
        empty_state("사분면 데이터 없음")
    else:
        # active_brands 필터 적용
        quad_df = quad_df[quad_df["brand_key"].isin(active_brands)]
        med_x = float(quad_df["리뷰 수"].median())
        med_y = float(quad_df["평균 평점"].median())

        fig_q = px.scatter(
            quad_df,
            x="리뷰 수", y="평균 평점",
            color="브랜드",
            color_discrete_map={BRANDS[b]["label"]: BRANDS[b]["color"] for b in BRAND_ORDER},
            text="카테고리",
            labels={"리뷰 수": "리뷰 수 (볼륨)", "평균 평점": "평균 평점 (만족도)"},
            title="브랜드 x 카테고리 — 리뷰 볼륨 vs 평점",
            hover_data={"브랜드": True, "카테고리": True, "리뷰 수": True, "평균 평점": True},
        )
        fig_q.add_hline(y=med_y, line_dash="dot", line_color="gray", opacity=0.5)
        fig_q.add_vline(x=med_x, line_dash="dot", line_color="gray", opacity=0.5)
        fig_q.update_traces(textposition="top center", textfont_size=10, marker_size=12)
        fig_q.update_layout(height=540, hovermode="closest")
        st.plotly_chart(fig_q, use_container_width=True)

st.divider()

# ─────────────────────────────────────────────────────────────
# 7. Action Recommendation — 상품 전략 요약
# ─────────────────────────────────────────────────────────────
st.subheader("Action Recommendation — 상품 전략 요약")
st.caption("BERTopic 신규 데이터(110M·22M·9.4K) + ABSA 6속성(29,070건) + 정형 분석 종합")

action_cols = st.columns(3)
actions = [
    {
        "title": "1순위 방어선 — 사이즈 표 정확화 + 형태 안정성",
        "color": "#C62828",
        "desc": (
            "<strong>저평점 9,445건의 83%</strong>가 핏(46.4%) + 품질/내구성(36.8%)에 집중.<br/>"
            "Top 부정 토픽: 교환·작다·반품(3,195건) → 보풀·세탁·양말(1,155건).<br/><br/>"
            "<strong>체형 사이즈 가이드 + 50회 세탁 후 형태 유지 인증</strong>을 PDP·정책에 명시 "
            "→ 초기 반품률·CS 부담 직접 절감"
        ),
        "evidence": "근거: dashboard_reviews_low.parquet 30토픽 + 저평점 aspect 분포",
    },
    {
        "title": "초기 SKU — 무채색 60% + 디자인 강점 카드",
        "color": "#1A1A1A",
        "desc": (
            "컬러 빈도 4브랜드 공통 Top3: <strong>블랙 · 네이비 · 그레이</strong>.<br/>"
            "FILA 디자인 P_ratio <strong>+0.611로 4브랜드 1위</strong> "
            "(룰루레몬 +0.519 / 젝시믹스 +0.405).<br/><br/>"
            "<strong>무채색 60% · 헤리티지 컬러 40%</strong> 라인업으로 디자인 강점을 "
            "PDP 비주얼·광고에서 전면 활용 — 차별화 첫 진입 카드"
        ),
        "evidence": "근거: 컬러 빈도 §5 + ABSA 디자인 P_ratio (phase_e 12,056건)",
    },
    {
        "title": "중장기 — 기능성 발화 점유 회복",
        "color": "#1565C0",
        "desc": (
            "BERTopic 발화 비중: 룰루레몬 기능성 <strong>46.7%</strong> vs "
            "FILA <strong>9.7%</strong> — 의류 기능성 담론 자체에 없음.<br/>"
            "ABSA에서도 기능성 +0.382로 4브랜드 최저.<br/><br/>"
            "<strong>쿨링 · 압박 · 통기성 R&amp;D 메시지</strong>를 "
            "PDP · 광고 카피 · 인플루언서 콘텐츠에 일관 노출, "
            "리뷰 토픽 점유율 <strong>30% 도달</strong>이 12개월 KPI"
        ),
        "evidence": "근거: dashboard_reviews_22M.parquet 브랜드×aspect_label + ABSA §4",
    },
]
for col, a in zip(action_cols, actions):
    with col:
        st.markdown(
            f"""<div style='border: 2px solid {a['color']}; border-radius: 8px;
            padding: 16px; height: 100%; background: {a['color']}11;'>
                <div style='color: {a['color']}; font-weight: 700; font-size: 15px; margin-top: 4px;'>
                    {a['title']}
                </div>
                <p style='font-size: 14px; margin-top: 12px; color: #222; line-height: 1.7;'>
                    {a['desc']}
                </p>
                <div style='font-size: 11px; color: #777; margin-top: 10px; font-style: italic;'>
                    {a['evidence']}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
