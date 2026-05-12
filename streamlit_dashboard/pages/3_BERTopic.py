"""
3_BERTopic.py — 고객의 목소리 (BERTopic 토픽 모델링, 2026-05-10 데이터)
==========================================================================

새 산출물 연결:
  - dashboard_reviews_110M.parquet  (전체 1.11M건)
  - dashboard_reviews_22M.parquet   (균형 샘플 213,972건)
  - dashboard_reviews_low.parquet   (저평점 9,448건, 별도 토픽 모델)
  - topic_dictionary.csv            (49개 토픽 메타)

※ 새 산출물에는 review_id가 없어 ABSA·기존 reviews와 join 불가.
   페이지 단독 분석 전용. (BERTopic × ABSA 히트맵은 4_ABSA.py에서
   `athleisure_bertopic.parquet` 별도 사용)
"""
from __future__ import annotations

import html as _html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import APP_TITLE, PAGE_LAYOUT, BRANDS, BRAND_ORDER, BRAND_COLORS, BERT_ASPECT_COLOR
from utils.data_loader import get_dashboard_reviews, get_topic_dictionary
from utils.session import init_session, mark_page_visited
from utils.exceptions import safe_block, empty_state
from components.filters import render_sidebar_filters
from components.page_header import render_page_intro

st.set_page_config(page_title=f"{APP_TITLE} — BERTopic", page_icon=None, **PAGE_LAYOUT)
init_session()
mark_page_visited("voc")

st.title("BERTopic")
st.caption("BERTopic 모델로 자동 추출된 핵심 토픽 — 49개 토픽, 6 aspect 그룹")
st.markdown(
    "<p style='font-size:12px; color:#888; margin:2px 0 2px;'>"
    "홈 &nbsp;›&nbsp; 상품/고객 전략 &nbsp;›&nbsp; <strong>BERTopic</strong> "
    "&nbsp;›&nbsp; ABSA &nbsp;›&nbsp; 포지셔닝</p>",
    unsafe_allow_html=True,
)
st.caption("리뷰 1,110,168건에서 추출된 49개 토픽을 6 aspect(사이즈·소재·기능·디자인 등)로 그룹화하여 살펴봅니다.")

render_page_intro(
    "리뷰에서 자동 추출된 49개 토픽으로 소비자가 실제로 무엇을 말하는지 "
    "aspect 그룹·키워드 단위로 확인하고, 4브랜드별 토픽 점유율과 저평점 토픽까지 비교합니다.",
    accent="#1565C0",
)

# 사이드바 — BERTopic 데이터에 없는 year/category/price 필터는 숨김
render_sidebar_filters(show_year=False, show_price=False, show_category=False)

# ── 데이터 스코프 토글 ───────────────────────────────────────
st.markdown("##### 분석 스코프")
_scope_col, _scope_desc = st.columns([1, 3])
with _scope_col:
    scope = st.radio(
        "데이터 스코프",
        options=["all", "22m", "low"],
        format_func=lambda s: {
            "all": "전체 (1.11M)",
            "22m": "균형 샘플 (213K)",
            "low": "저평점 리뷰 (9.4K)",
        }[s],
        index=0,
        key="bert_scope",
        label_visibility="collapsed",
    )
with _scope_desc:
    if scope == "all":
        st.info("**전체 모드** — 1,110,168건 / 절대 빈도 비교에 적합")
    elif scope == "22m":
        st.warning("**균형 샘플 모드** — 213,972건 / 4브랜드 균형 표본 / 일반 분석 권장")
    else:
        st.warning(
            "**저평점 모드** — 평점 1~2점 9,448건만 / 부정 리뷰의 핵심 클러스터 식별에 특화")

# ── 데이터 로드 ──────────────────────────────────────────────
with safe_block("BERTopic 데이터 로드"):
    df_raw = get_dashboard_reviews(scope=scope)      # 원본 데이터 로드
    topic_dict = get_topic_dictionary(scope=scope)   # 49(또는 30)개 토픽 메타

# 사이드바 필터 적용 (brand·rating만 — BERTopic 데이터에는 year/category/price 컬럼 없음)
if not df_raw.empty:
    _b_raw = st.session_state.get("brands") or list(BRAND_ORDER)
    f_brands = [_b_raw] if isinstance(_b_raw, str) else list(_b_raw)
    if not f_brands:
        f_brands = list(BRAND_ORDER)
    f_rating = st.session_state.get("rating_sel", "전체")

    mask = df_raw["brand"].isin(f_brands)
    if f_rating != "전체":
        try:
            mask &= (df_raw["rating"] == int(f_rating))
        except (ValueError, TypeError):
            pass
    df = df_raw[mask].copy()
else:
    df = df_raw

if df.empty:
    empty_state("BERTopic 데이터 부재", "dashboard_reviews_*.parquet 파일을 확인해 주세요.")
    st.stop()

# 스코프별 KPI 한 줄
_kpi_cols = st.columns(4)
with _kpi_cols[0]:
    st.metric("리뷰 수", f"{len(df):,}")
with _kpi_cols[1]:
    st.metric("토픽 수", f"{df['topic'].nunique():,}")
with _kpi_cols[2]:
    st.metric("브랜드 수", f"{df['brand'].nunique()}")
with _kpi_cols[3]:
    if "rating" in df.columns:
        st.metric("평균 평점", f"{df['rating'].mean():.2f}")

st.divider()


# ─────────────────────────────────────────────────────────────
# Aspect 그룹별 분포 (저평점은 topic_name에서 추출, 그 외는 dict 활용)
# ─────────────────────────────────────────────────────────────
st.subheader("속성 그룹별 토픽 분포")
st.caption("리뷰가 어느 속성에 쏠려 있는지 — 6개 속성 그룹 집계")

with safe_block("Aspect 분포"):
    # aspect_label은 data_loader가 이미 한글로 매핑 (BERT_ASPECT_KR + topic_name 폴백)
    df_aspect = df[df["topic"] >= 0].copy() if "topic" in df.columns else df.copy()
    df_aspect = df_aspect[df_aspect["aspect_label"] != "노이즈"]

    agg = df_aspect.groupby("aspect_label").size().rename("리뷰 수").reset_index()
    agg = agg.sort_values("리뷰 수", ascending=False)
    agg["비율"] = agg["리뷰 수"] / agg["리뷰 수"].sum()

    fig_aspect = px.bar(
        agg, x="aspect_label", y="리뷰 수",
        text=agg["비율"].apply(lambda v: f"{v:.1%}"),
        labels={"aspect_label": "속성 그룹", "리뷰 수": "리뷰 수(건)"},
        color="aspect_label",
        color_discrete_map=BERT_ASPECT_COLOR,
    )
    fig_aspect.update_traces(textposition="outside", textfont_size=12, marker_line_width=0)
    fig_aspect.update_layout(
        height=320,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis_tickangle=0,
        xaxis_title=""
    )
    st.plotly_chart(fig_aspect, use_container_width=True)

st.divider()


# ─────────────────────────────────────────────────────────────
# 토픽 트리맵 + 토픽 카드
# ─────────────────────────────────────────────────────────────
col_tm, col_card = st.columns([2, 1])

with col_tm:
    st.subheader("토픽 트리맵")
    st.caption("토픽 면적 = 리뷰 수 / aspect 그룹별 색상")

    with safe_block("토픽 트리맵"):
        # 토픽별 size 집계 (aspect_label은 행마다 일정하므로 first로 가져오기)
        topic_size = (
            df[df["topic"] >= 0]
            .groupby(["topic", "topic_name"], as_index=False)
            .agg(n=("brand", "size"), aspect_label=("aspect_label", "first"))
        )
        topic_size["aspect_label"] = topic_size["aspect_label"].fillna("기타")
        topic_size = topic_size[topic_size["aspect_label"] != "노이즈"]

        fig_tree = px.treemap(
            topic_size,
            path=["aspect_label", "topic_name"],
            values="n",
            color="aspect_label",
            color_discrete_map=BERT_ASPECT_COLOR,
            hover_data={"n": ":,"},
        )
        fig_tree.update_traces(
            textinfo="label+value+percent parent",
            textfont_size=11,
            marker=dict(line=dict(color="white", width=1)),
        )
        fig_tree.update_layout(
            height=520,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_tree, use_container_width=True)

with col_card:
    st.subheader("Top 8 토픽")
    st.caption("리뷰 수 기준 상위 8개")

    top8 = (
        df.groupby(["topic", "topic_name"]).size().rename("n").reset_index()
    )
    top8 = top8[top8["topic"] >= 0].nlargest(8, "n")

    # 키워드 매핑 — topic_aspect_mapping.parquet의 keywords_top5 활용
    kw_map = {}
    if not topic_dict.empty and "keywords_top5" in topic_dict.columns:
        kw_map = topic_dict.set_index("topic_id")["keywords_top5"].to_dict()

    for _, row in top8.iterrows():
        kws = kw_map.get(row["topic"], "")
        name_safe = _html.escape(str(row["topic_name"]))
        kws_safe  = _html.escape(str(kws)) if kws else ""
        kws_div   = (f'<div style="font-size:11px;color:#444;margin-top:4px;">'
                     f'키워드: {kws_safe}</div>') if kws_safe else ""
        st.markdown(
            f"<div style='border-left:3px solid #1565C0;padding:8px 12px;"
            f"margin-bottom:8px;background:#F0F4FF;border-radius:0 4px 4px 0;'>"
            f"<div style='font-weight:600;font-size:13px;'>{name_safe}</div>"
            f"<div style='font-size:11px;color:#666;margin-top:3px;'>{int(row['n']):,} 리뷰</div>"
            f"{kws_div}</div>",
            unsafe_allow_html=True,
        )

st.divider()

# 💡 브랜드별 그라데이션 컬러 팔레트 정의
BRAND_GRADIENTS = {
    "FILA": ["#001A33", "#002A54", "#004080", "#0059B3", "#3385FF", "#80B3FF"], # 네이비 계열
    "젝시믹스": ["#800000", "#B30000", "#D70100", "#FF3333", "#FF6666", "#FF9999"], # 레드 계열
    "룰루레몬": ["#000000", "#1A1A1A", "#333333", "#4D4D4D", "#666666", "#999999"], # 블랙/그레이 계열
    "안다르": ["#B3A693", "#D6CDC0", "#E5DDD1", "#F2ECE4", "#FAF8F5", "#FFFFFF"]  # 베이지/웜톤 계열
}

# ─────────────────────────────────────────────────────────────
# 브랜드별 속성 그룹 비중 (파이 차트)
# ─────────────────────────────────────────────────────────────
st.subheader("브랜드별 속성 그룹 비중")
st.caption("4개 브랜드 리뷰에서 어떤 속성 그룹이 가장 많이 언급되는지 — 파이 차트로 비교")

with safe_block("브랜드별 속성 그룹 파이"):
    brands_present = [b for b in BRAND_ORDER if b in df["brand"].unique()]
    if not brands_present:
        empty_state("브랜드 매칭 결과 없음")
    else:
        # data_loader가 이미 aspect_label 매핑 완료 — 노이즈만 제외
        df_asp = df[df["aspect_label"] != "노이즈"].copy()

        pie_cols = st.columns(len(brands_present))
        for col_ui, brand in zip(pie_cols, brands_present):
            brand_sub = df_asp[df_asp["brand"] == brand]
            if brand_sub.empty:
                continue
            asp_counts = (
                brand_sub.groupby("aspect_label").size()
                .reset_index(name="리뷰 수")
                .sort_values("리뷰 수", ascending=False)
            )
            with col_ui:
                color_seq = BRAND_GRADIENTS.get(brand, px.colors.sequential.Blues)

                fig_pie = px.pie(
                    asp_counts,
                    values="리뷰 수",
                    names="aspect_label",
                    title=BRANDS[brand]["label"],
                    color_discrete_sequence=color_seq,
                    hole=0.3,
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent+label",
                    textfont_size=13,
                )
                fig_pie.update_layout(
                    height=320,
                    margin=dict(t=40, b=20, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        # 속성별 포함 토픽 안내 텍스트
        st.markdown("---")
        if "aspect_label" in df_asp.columns and "topic_name" in df_asp.columns:
            for asp_label in sorted(df_asp["aspect_label"].dropna().unique()):
                top_topics_list = (
                    df_asp[df_asp["aspect_label"] == asp_label]
                    .groupby("topic_name").size()
                    .nlargest(5).index.tolist()
                )
                if top_topics_list:
                    st.caption(f"※ **{asp_label}** 속성 포함 토픽: {', '.join(top_topics_list)} 등")

st.divider()


# ─────────────────────────────────────────────────────────────
# 토픽 드릴다운 — 선택 토픽 → 샘플 리뷰
# ─────────────────────────────────────────────────────────────
st.subheader("토픽 드릴다운")
st.caption("선택한 토픽에서 어떤 리뷰가 모였는지 — 평점·브랜드별 샘플 확인")

with safe_block("드릴다운"):
    topic_options = (
        df[df["topic"] >= 0]
        .groupby(["topic", "topic_name"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False)
    )
    topic_label_to_id = dict(zip(topic_options["topic_name"], topic_options["topic"]))
    sel_topic_name = st.selectbox(
        "토픽 선택",
        options=topic_options["topic_name"].tolist(),
        key="bert_drill_topic",
    )

    if sel_topic_name:
        tid = topic_label_to_id[sel_topic_name]
        sample = df[df["topic"] == tid].copy()

        sample_kpi = st.columns(4)
        with sample_kpi[0]:
            st.metric("토픽 리뷰 수", f"{len(sample):,}")
        with sample_kpi[1]:
            st.metric("평균 평점", f"{sample['rating'].mean():.2f}")
        with sample_kpi[2]:
            st.metric("브랜드 수", f"{sample['brand'].nunique()}")
        with sample_kpi[3]:
            top_brand = sample["brand"].mode().iloc[0] if not sample.empty else "-"
            st.metric("최다 브랜드", BRANDS.get(top_brand, {}).get("label", top_brand))

        rc1, rc2 = st.columns([1, 1])
        with rc1:
            sel_brand = st.multiselect(
                "브랜드 필터",
                options=sorted(sample["brand"].unique().tolist()),
                default=sorted(sample["brand"].unique().tolist()),
                key="bert_drill_brand",
            )
        with rc2:
            sel_rating = st.multiselect(
                "평점 필터",
                options=sorted(sample["rating"].unique().tolist()),
                default=sorted(sample["rating"].unique().tolist()),
                key="bert_drill_rating",
            )

        sample_f = sample[
            sample["brand"].isin(sel_brand) & sample["rating"].isin(sel_rating)
        ]
        st.caption(f"필터 적용 후 {len(sample_f):,}건 / 표시 상위 30건")
        if sample_f.empty:
            empty_state("필터 결과 없음")
        else:
            view_cols = ["brand", "rating", "topic_name", "content"]
            view = sample_f[view_cols].rename(columns={
                "brand": "브랜드", "rating": "평점",
                "topic_name": "토픽명", "content": "리뷰 원문",
            }).head(30)
            st.dataframe(view, use_container_width=True, hide_index=True)
