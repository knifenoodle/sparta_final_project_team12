"""
charts.py — Plotly 기반 공용 차트 헬퍼
========================================

라이브러리 선택 근거:
- Plotly: 인터랙티브 호버, 줌/팬, 범례 토글이 모두 기본 제공.
  C-Level 보고에서 청중이 직접 데이터를 탐색하는 시연이 가능.
- Altair: 선언적 문법은 깔끔하나 호버 커스터마이징과 산점도 어노테이션이 제한적.
- 결론: 핵심 시각화는 Plotly, 단순 분포는 st.bar_chart 로 분담.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import BRAND_COLORS, ASPECT_LABELS, SENTIMENT_COLOR


def rating_distribution(reviews: pd.DataFrame) -> go.Figure:
    """브랜드 × 평점 분포 (stacked bar, 100% 정규화)."""
    if reviews.empty or "rating" not in reviews.columns:
        return _empty_fig("평점 데이터 없음")
    ct = (reviews.groupby(["brand", "rating"]).size()
          .groupby(level=0).apply(lambda s: s / s.sum())
          .rename("ratio").reset_index())
    fig = px.bar(
        ct, x="brand", y="ratio", color="rating",
        color_continuous_scale="RdYlGn",
        labels={"ratio": "비율", "brand": "브랜드", "rating": "평점"},
        title="브랜드별 평점 분포 (정규화)",
    )
    fig.update_layout(barmode="stack", yaxis_tickformat=".0%", height=380)
    return fig


def review_volume_timeline(reviews: pd.DataFrame) -> go.Figure:
    """월별 리뷰 볼륨 추이 (브랜드 라인)."""
    if reviews.empty or "review_date" not in reviews.columns:
        return _empty_fig("타임라인 데이터 없음")
    df = reviews.copy()
    df["yyyymm"] = df["review_date"].dt.to_period("M").astype(str)
    agg = df.groupby(["yyyymm", "brand"]).size().reset_index(name="n")
    fig = px.line(
        agg, x="yyyymm", y="n", color="brand",
        color_discrete_map=BRAND_COLORS,
        markers=True,
        labels={"yyyymm": "연-월", "n": "리뷰 수"},
        title="월별 리뷰 볼륨 추이",
    )
    fig.update_layout(height=380, hovermode="x unified")
    return fig


def aspect_polarity_grouped_bar(polarity_df: pd.DataFrame) -> go.Figure:
    """브랜드 × 6속성 P 비율 그룹 막대.

    polarity_df: long-format [brand, aspect, P_ratio, N_ratio, X_ratio]
    """
    if polarity_df.empty:
        return _empty_fig("ABSA 결과 없음")
    df = polarity_df.copy()
    df["aspect_label"] = df["aspect"].map(ASPECT_LABELS)
    fig = px.bar(
        df, x="aspect_label", y="P_ratio", color="brand",
        color_discrete_map=BRAND_COLORS, barmode="group",
        labels={"aspect_label": "속성", "P_ratio": "긍정(P) 비율", "brand": "브랜드"},
        title="브랜드 × 6속성 긍정 비율",
        hover_data={"N_ratio": ":.2%", "X_ratio": ":.2%", "n_reviews": True},
    )
    fig.update_layout(yaxis_tickformat=".0%", height=440)
    return fig


def aspect_polarity_diverging_bar(polarity_df: pd.DataFrame, brand: str) -> go.Figure:
    """단일 브랜드 — P/N 양쪽 발산 막대 (gain/loss 시각화)."""
    if polarity_df.empty:
        return _empty_fig("데이터 없음")
    sub = polarity_df[polarity_df["brand"] == brand].copy()
    if sub.empty:
        return _empty_fig(f"{brand} ABSA 결과 없음")
    sub["aspect_label"] = sub["aspect"].map(ASPECT_LABELS)
    sub["N_neg"] = -sub["N_ratio"]
    fig = go.Figure()
    fig.add_bar(name="긍정 P", y=sub["aspect_label"], x=sub["P_ratio"],
                orientation="h", marker_color=SENTIMENT_COLOR["P"],
                hovertemplate="P: %{x:.1%}<extra></extra>")
    fig.add_bar(name="부정 N", y=sub["aspect_label"], x=sub["N_neg"],
                orientation="h", marker_color=SENTIMENT_COLOR["N"],
                hovertemplate="N: %{customdata:.1%}<extra></extra>",
                customdata=sub["N_ratio"])
    fig.update_layout(
        title=f"{brand} — 속성별 긍/부정 균형",
        barmode="overlay", height=380, xaxis_tickformat=".0%",
        xaxis_title="비율 (← 부정 | 긍정 →)",
    )
    return fig


def topic_keyword_treemap(topic_meta: pd.DataFrame) -> go.Figure:
    """토픽 × n_reviews 트리맵 (라벨에 비율 표시)."""
    if topic_meta.empty:
        return _empty_fig("토픽 메타 없음")
    df = topic_meta[topic_meta["topic_id"] >= 0].copy()
    df["keywords_str"] = df["keywords"].apply(lambda kws: ", ".join(kws[:6]))
    total = df["n_reviews"].sum() or 1
    df["share"] = df["n_reviews"] / total
    fig = px.treemap(
        df,
        path=["topic_name"],
        values="n_reviews",
        color="n_reviews",
        color_continuous_scale="Tealgrn",
        custom_data=["keywords_str", "share"],
        title="토픽 점유율",
    )
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{customdata[1]:.1%} (%{value:,}건)",
        textposition="middle center",
        textfont_size=13,
        hovertemplate=(
            "<b>%{label}</b><br>"
            "리뷰 %{value:,}건 (%{customdata[1]:.1%})<br>"
            "키워드: %{customdata[0]}<extra></extra>"
        ),
    )
    fig.update_layout(height=480)
    return fig


def keyword_centrality_bar(sna_df: pd.DataFrame, brand: str, top_n: int = 15) -> go.Figure:
    """SNA 중심성 상위 키워드."""
    if sna_df.empty:
        return _empty_fig("SNA 데이터 없음")
    sub = sna_df[sna_df["brand"] == brand].nlargest(top_n, "centrality")
    if sub.empty:
        return _empty_fig(f"{brand} 키워드 없음")
    fig = px.bar(
        sub.sort_values("centrality"),
        x="centrality", y="keyword", orientation="h",
        color="polarity",
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        labels={"centrality": "중심성", "keyword": "키워드", "polarity": "감성"},
        title=f"{brand} 핵심 키워드 (상위 {top_n})",
        hover_data={"frequency": True},
    )
    fig.update_layout(height=440, coloraxis_colorbar=dict(title="P↔N"))
    return fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=16, color="gray"),
    )
    fig.update_layout(height=320, xaxis_visible=False, yaxis_visible=False)
    return fig
