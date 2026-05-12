"""
4_ABSA.py — 브랜드 속성 평가 (EXAONE ABSA 6속성)
=================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import APP_TITLE, PAGE_LAYOUT, BRAND_ORDER, BRANDS, PATHS, ASPECT_LABELS, ASPECTS
from utils.data_loader import compute_aspect_polarity, filters_to_hash
from utils.session import init_session, get_filters, mark_page_visited
from utils.exceptions import safe_block, empty_state, warn_using_dummy
from components.filters import render_sidebar_filters
from components.charts import aspect_polarity_diverging_bar
from components.page_header import render_page_intro


st.set_page_config(page_title=f"{APP_TITLE} — ABSA", page_icon=None, **PAGE_LAYOUT)
init_session()
mark_page_visited("absa")

st.title("ABSA")
st.caption("EXAONE 3.5 기반 6속성 P/N/X 분석 결과")
st.markdown(
    "<p style='font-size:12px; color:#888; margin:2px 0 2px;'>"
    "홈 &nbsp;›&nbsp; 상품/고객 전략 &nbsp;›&nbsp; BERTopic &nbsp;›&nbsp; "
    "<strong>ABSA</strong> &nbsp;›&nbsp; 포지셔닝</p>",
    unsafe_allow_html=True,
)
st.caption("BERTopic이 발굴한 핵심 토픽을 6가지 속성 단위의 감성(P/N/X)으로 정량화합니다.")

render_page_intro(
    "EXAONE 3.5로 추론한 6속성(핏·소재·기능·디자인·브랜드·가격) P/N/X 결과로 "
    "브랜드별 강점·약점을 정량 비교하고, 토픽·시계열 단위로 미시 인사이트를 도출합니다.",
    accent="#D4000F",
)

with st.expander("지표 정의 — P/N/X 비율의 분모는 무엇인가요?"):
    st.markdown(
        """
- **P_ratio (긍정 비율)** = 해당 속성에서 **P로 분류된 리뷰 수 / 해당 속성이 언급된 전체 리뷰 수**
- **N_ratio (부정 비율)** = 해당 속성에서 **N으로 분류된 리뷰 수 / 해당 속성이 언급된 전체 리뷰 수**
- **X_ratio (미언급/중립 비율)** = `1 − P_ratio − N_ratio`
- 분모는 **해당 속성이 언급된 리뷰**입니다 (전체 리뷰가 아닙니다).
  - 예: 기능성 P_ratio 60% = "기능성을 언급한 리뷰 100건 중 60건이 긍정"
- X_ratio가 높은 속성은 **"해당 속성에 대한 언급 자체가 적다"**는 뜻이며,
  P/N 비율 해석 시 분모 크기(언급 수)를 함께 봐야 합니다.
        """
    )

_absa_available = PATHS["absa"].exists()
if not _absa_available:
    warn_using_dummy("ABSA 감성 분석")

render_sidebar_filters()
filters = get_filters()
fh = filters_to_hash(filters)

# ── 분석 모드 (균형 비교 고정) ───────────────
st.markdown("##### 분석 모드")
analysis_mode = "balanced"  # 하위 코드 에러 방지를 위해 변수 고정

st.info(
    "**균형 비교** — 층화 표본(브랜드당 3,014건)으로 4브랜드 통계적 동등 비교. "
    "신뢰구간이 모든 브랜드 동일 → 페어 비교의 정합성 확보."
)

# ── 데이터 ─────────────────────────────────────────────────
with safe_block("ABSA 데이터 로드"):
    polarity = compute_aspect_polarity(filters_hash=fh, sample_mode=analysis_mode)

if polarity.empty:
    empty_state("ABSA 결과 없음", "EXAONE 추론 완료 후 자동 갱신")
    st.stop()

# ── 브랜드 부정 비율 KPI 게이지 ──────────────────────────────
st.markdown("##### 브랜드별 부정 비율 한눈 게이지")
st.caption("6속성 평균 N_ratio — 막대 길이·색상이 진할수록 위험 신호")
with safe_block("브랜드 부정 비율"):
    _neg_rows = []
    for _brand in BRAND_ORDER:
        _sub = polarity[polarity["brand"] == _brand]
        if _sub.empty:
            continue
        _neg_rows.append({
            "brand": BRANDS[_brand]["label"],
            "N_ratio_avg": float(_sub["N_ratio"].mean()),
        })
    if _neg_rows:
        _neg_df = pd.DataFrame(_neg_rows).sort_values("N_ratio_avg", ascending=True)
        _fig_neg = px.bar(
            _neg_df, x="N_ratio_avg", y="brand",
            orientation="h",
            color="N_ratio_avg",
            color_continuous_scale=[
                (0.0, "#5798D9"), (0.3, "#002A54"), (0.7, "#8A73E5"), (1.0, "#B71C1C"),
            ],
            range_color=[0.0, 0.30],
            text=_neg_df["N_ratio_avg"].apply(lambda v: f"{v:.1%}"),
            labels={"N_ratio_avg": "평균 N_ratio (6속성 평균)", "brand": ""},
        )
        _fig_neg.update_traces(textposition="outside", textfont=dict(size=12))
        _fig_neg.update_layout(
            height=200,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(tickformat=".0%", range=[0.0, max(0.05, _neg_df["N_ratio_avg"].max() * 1.4)]),
            coloraxis_showscale=False,
        )
        st.plotly_chart(_fig_neg, use_container_width=True)

st.divider()

# ── 레이더 차트 — 4브랜드 × 6속성 ────────────────────────────
# st.subheader("브랜드 × 6속성 레이더 차트 — 포지셔닝 비교")

# with safe_block("레이더 차트"):
#     aspect_keys = [a["key"] for a in ASPECTS]
#     labels = [ASPECT_LABELS[k] for k in aspect_keys]
#     labels_closed = labels + [labels[0]]

#     fig_radar = go.Figure()
#     for brand in BRAND_ORDER:
#         sub = polarity[polarity["brand"] == brand]
#         if sub.empty:
#             continue
#         vals = []
#         for asp in aspect_keys:
#             row = sub[sub["aspect"] == asp]
#             vals.append(float(row["P_ratio"].iloc[0]) if not row.empty else 0.0)
#         vals_closed = vals + [vals[0]]

#         color = BRANDS[brand]["color"]
#         fig_radar.add_trace(go.Scatterpolar(
#             r=vals_closed,
#             theta=labels_closed,
#             fill="toself",
#             fillcolor=color + "2E",  # ~18% opacity
#             line=dict(color=color, width=2.5),
#             name=BRANDS[brand]["label"],
#             hovertemplate="%{theta}: %{r:.1%}<extra>" + BRANDS[brand]["label"] + "</extra>",
#         ))

#     fig_radar.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1],
#                 tickformat=".0%",
#                 tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
#                 tickfont=dict(size=10),
#                 gridcolor="#D0D8EC",
#             ),
#             angularaxis=dict(
#                 tickfont=dict(size=12),
#                 gridcolor="#D0D8EC",
#             ),
#             bgcolor="#F8FAFF",
#         ),
#         height=540,
#         showlegend=True,
#         legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
#         paper_bgcolor="#FFFFFF",
#     )
#     st.plotly_chart(fig_radar, use_container_width=True)

# # 1. 차트 렌더링 코드 바로 위에 헥스 변환 함수 추가
# def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
#     """6자리 Hex 코드를 rgba(r, g, b, alpha) 포맷으로 변환"""
#     hex_color = hex_color.lstrip('#')
#     if len(hex_color) == 6:
#         r = int(hex_color[0:2], 16)
#         g = int(hex_color[2:4], 16)
#         b = int(hex_color[4:6], 16)
#         return f"rgba({r}, {g}, {b}, {alpha})"
#     return f"#{hex_color}"

# # 2. 레이더 차트 본문
# st.subheader("브랜드 × 6속성 레이더 차트 — 포지셔닝 비교")

# with safe_block("레이더 차트"):
#     aspect_keys = [a["key"] for a in ASPECTS]
#     labels = [ASPECT_LABELS[k] for k in aspect_keys]
#     labels_closed = labels + [labels[0]]

#     fig_radar = go.Figure()
#     for brand in BRAND_ORDER:
#         sub = polarity[polarity["brand"] == brand]
#         if sub.empty:
#             continue
#         vals = []
#         for asp in aspect_keys:
#             row = sub[sub["aspect"] == asp]
#             vals.append(float(row["P_ratio"].iloc[0]) if not row.empty else 0.0)
#         vals_closed = vals + [vals[0]]

#         color = BRANDS[brand]["color"]
#         # 수정됨: 투명도 18%가 적용된 rgba 색상 적용
#         fill_color_with_alpha = hex_to_rgba(color, alpha=0.18)

#         fig_radar.add_trace(go.Scatterpolar(
#             r=vals_closed,
#             theta=labels_closed,
#             fill="toself",
#             fillcolor=fill_color_with_alpha,  # 에러 해결!
#             line=dict(color=color, width=2.5),
#             name=BRANDS[brand]["label"],
#             hovertemplate="%{theta}: %{r:.1%}<extra>" + BRANDS[brand]["label"] + "</extra>",
#         ))

#     fig_radar.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1],
#                 tickformat=".0%",
#                 tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
#                 tickfont=dict(size=10),
#                 gridcolor="#D0D8EC",
#             ),
#             angularaxis=dict(
#                 tickfont=dict(size=12),
#                 gridcolor="#D0D8EC",
#             ),
#             bgcolor="#F8FAFF",
#         ),
#         height=540,
#         showlegend=True,
#         legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
#         paper_bgcolor="#FFFFFF",
#     )
#     st.plotly_chart(fig_radar, use_container_width=True)

# ── [E] 브랜드 × 속성 Sentiment Score 히트맵 ─────────────────
st.subheader("브랜드 × 속성 감성 점수 히트맵")
st.caption(
    "감성 점수(Sentiment Score) = 긍정 비율(P) − 부정 비율(N)  ∈ [-1, 1]. "
    "0 = 중립(P=N) / +0.5 = 강한 긍정 우세 / -0.5 = 강한 부정 우세."
)


with safe_block("Sentiment Score 히트맵"):
    _aspect_order  = [a["key"] for a in ASPECTS]
    _aspect_labels = [ASPECT_LABELS[k] for k in _aspect_order]
    _brand_order_f = [b for b in BRAND_ORDER if b in polarity["brand"].unique()]
    _brand_labels  = [BRANDS[b]["label"] for b in _brand_order_f]

    # Sentiment Score = P − N (절대 스케일, [-1, 1])
    _polarity_ss = polarity.copy()
    _polarity_ss["sentiment_score"] = _polarity_ss["P_ratio"] - _polarity_ss["N_ratio"]

    _pivot = (
        _polarity_ss[_polarity_ss["brand"].isin(_brand_order_f)]
        .pivot(index="brand", columns="aspect", values="sentiment_score")
        .reindex(index=_brand_order_f, columns=_aspect_order)
    )

    _hm_z = _pivot.round(2).values
    _fig_hm = px.imshow(
        _hm_z,
        x=_aspect_labels,
        y=_brand_labels,
        color_continuous_scale="RdBu",
        zmin=-0.5, zmax=0.5,
        color_continuous_midpoint=0.0,
        aspect="auto",
        labels={"color": "감성 점수"},
    )
    _fig_hm.update_layout(
        height=260,
        margin=dict(t=20, b=20, l=10, r=10),
        coloraxis_colorbar=dict(
            tickformat="+.2f",
            title="감성 점수<br>(P−N)",
            len=0.9,
            tickvals=[-0.5, -0.25, 0.0, 0.25, 0.5],
        ),
        xaxis=dict(side="bottom"),
    )
    _fig_hm.update_traces(
        text=[[f"{v:.2f}" for v in row] for row in _hm_z],
        texttemplate="%{text}",
        textfont=dict(size=12),
    )
    st.plotly_chart(_fig_hm, use_container_width=True)
    st.caption(
        "음수 셀(빨강) = 해당 속성에서 부정 우세 / 양수 셀(파랑) = 긍정 우세 / "
        "0 부근(노랑) = 양극 균형. P_ratio 단독보다 약점 식별에 유리"
    )

st.divider()

# ── BERTopic × ABSA 교차 히트맵 ─────────────────────────────
st.subheader("BERTopic × ABSA 교차 히트맵")
st.caption(
    "토픽 × 6속성 — 토픽별 어떤 속성이 가장 긍정/부정인지. "
    "Sentiment Score = P_ratio − N_ratio (∈ [-1, 1])"
)

with safe_block("BERTopic × ABSA 데이터 조인"):
    from utils.data_loader import get_topics, get_topic_meta, get_reviews, get_absa

    _topics_df = get_topics()
    _meta_df   = get_topic_meta()
    _absa_df   = get_absa(sample_mode=analysis_mode)

    if "topic_id" not in _topics_df.columns:
        for _old in ("Topic", "topic"):
            if _old in _topics_df.columns:
                _topics_df = _topics_df.rename(columns={_old: "topic_id"})
                break

    if (_topics_df.empty or _absa_df.empty
            or "topic_id" not in _topics_df.columns
            or "review_id" not in _topics_df.columns):
        empty_state("BERTopic × ABSA 조인 불가", "토픽 또는 ABSA 데이터 부재")
    else:
        _tdf = _topics_df[_topics_df["topic_id"] >= 0][["review_id", "topic_id"]]
        _aspect_cols = [k for k in [a["key"] for a in ASPECTS] if k in _absa_df.columns]
        _absa_join = _absa_df[["review_id"] + _aspect_cols].copy()
        _merged = _tdf.merge(_absa_join, on="review_id", how="inner")

        if _merged.empty:
            empty_state("매칭 결과 없음")
        else:
            _MIN_CELL = 30
            _rows = []
            for _tid, _sub in _merged.groupby("topic_id"):
                if len(_sub) < _MIN_CELL:
                    continue
                _row = {"topic_id": int(_tid), "n": len(_sub)}
                for _asp in _aspect_cols:
                    _counts = _sub[_asp].value_counts(normalize=True)
                    _row[_asp] = float(_counts.get("P", 0.0)) - float(_counts.get("N", 0.0))
                _rows.append(_row)

            if not _rows:
                empty_state("표본 30건 이상 토픽 없음")
            else:
                _ta_df = pd.DataFrame(_rows)
                if (not _meta_df.empty and "topic_id" in _meta_df.columns
                        and "topic_name" in _meta_df.columns):
                    _ta_df = _ta_df.merge(
                        _meta_df[["topic_id", "topic_name"]].drop_duplicates(),
                        on="topic_id", how="left",
                    )
                _ta_df["topic_name"] = _ta_df.get("topic_name", _ta_df["topic_id"].astype(str))
                _ta_df["topic_name"] = _ta_df["topic_name"].fillna(_ta_df["topic_id"].astype(str))

                _ta_top = _ta_df.nlargest(15, "n").sort_values("n", ascending=False).reset_index(drop=True)
                _ta_top["display_label"] = _ta_top.apply(
                    lambda r: f"{r['topic_name']} (n={int(r['n']):,})", axis=1
                )

                _pivot_ta = _ta_top.set_index("display_label")[_aspect_cols]
                _pivot_ta.columns = [ASPECT_LABELS[k] for k in _aspect_cols]

                _ta_z = _pivot_ta.round(2).values
                _fig_ta = px.imshow(
                    _ta_z,
                    x=list(_pivot_ta.columns),
                    y=list(_pivot_ta.index),
                    color_continuous_scale="RdBu",
                    zmin=-0.5, zmax=0.5,
                    aspect="auto",
                    labels={"color": "감성 점수"},
                )
                _fig_ta.update_layout(
                    height=max(360, 32 * len(_pivot_ta)),
                    margin=dict(t=20, b=20, l=10, r=10),
                    coloraxis_colorbar=dict(title="감성 점수", len=0.9),
                )
                _fig_ta.update_traces(
                    text=[[f"{v:.2f}" for v in row] for row in _ta_z],
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                )
                st.plotly_chart(_fig_ta, use_container_width=True)
                st.caption(
                    f"표본 30건 미만 토픽 제외 / 상위 {len(_pivot_ta)}개 토픽 (표본 큰 순). "
                    "파랑 = 속성 긍정 우세, 빨강 = 속성 부정 우세"
                )

st.divider()

# ── 시계열 부정 트렌드 ──────────────────────────────────────
st.subheader("시계열 부정 트렌드")
st.caption(
    "월별 부정 비율(N_ratio) 추이 — 위기·개선 시그널 모니터링. "
    "월별 30건 미만 셀은 제외 (graceful degradation)"
)

with safe_block("시계열 부정 데이터 조인"):
    _reviews_ts = get_reviews(columns=("review_id", "brand", "review_date"))
    _absa_ts    = get_absa(sample_mode=analysis_mode)

    if _reviews_ts.empty or _absa_ts.empty:
        empty_state("시계열 데이터 부재")
    else:
        _aspect_cols_ts = [k for k in [a["key"] for a in ASPECTS] if k in _absa_ts.columns]
        _absa_ts_join = _absa_ts[["review_id"] + _aspect_cols_ts].copy()

        _df_ts = _reviews_ts.merge(_absa_ts_join, on="review_id", how="inner")
        _df_ts["review_date"] = pd.to_datetime(_df_ts["review_date"], errors="coerce")
        _df_ts = _df_ts.dropna(subset=["review_date"])
        _df_ts["ym"] = _df_ts["review_date"].dt.to_period("M").astype(str)

        _ts_aspect = st.radio(
            "속성",
            options=["전체"] + _aspect_cols_ts,
            format_func=lambda k: "전체 6속성 평균" if k == "전체" else ASPECT_LABELS[k],
            horizontal=True,
            key="ts_aspect",
        )

        _MIN_TS = 30
        _ts_rows = []
        for (_brand, _ym), _sub in _df_ts.groupby(["brand", "ym"]):
            if len(_sub) < _MIN_TS:
                continue
            if _ts_aspect == "전체":
                _n_vals = [
                    float(_sub[_a].value_counts(normalize=True).get("N", 0.0))
                    for _a in _aspect_cols_ts
                ]
                _n_ratio = sum(_n_vals) / len(_n_vals) if _n_vals else 0.0
            else:
                _counts = _sub[_ts_aspect].value_counts(normalize=True)
                _n_ratio = float(_counts.get("N", 0.0))
            _ts_rows.append({
                "brand": _brand, "ym": _ym,
                "N_ratio": _n_ratio, "n": len(_sub),
            })

        if not _ts_rows:
            empty_state("월 30건 이상 표본 없음")
        else:
            _ts_df = pd.DataFrame(_ts_rows).sort_values(["brand", "ym"])
            _color_map_ts = {b: BRANDS[b]["color"] for b in BRANDS}
            _fig_ts = px.line(
                _ts_df, x="ym", y="N_ratio", color="brand",
                color_discrete_map=_color_map_ts,
                markers=True,
                hover_data={"n": True, "N_ratio": ":.2%", "ym": False},
                labels={"ym": "월", "N_ratio": "N_ratio (부정 비율)", "brand": "브랜드"},
            )
            _fig_ts.update_layout(
                height=420,
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(tickformat=".0%"),
                legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
            )
            st.plotly_chart(_fig_ts, use_container_width=True)
            # _aspect_label_ts = "6속성 평균" if _ts_aspect == "전체" else ASPECT_LABELS[_ts_aspect]
            # st.caption(
            #     f"속성: {_aspect_label_ts} / 표시 월 수: {_ts_df['ym'].nunique()} / "
            #     f"표시 셀: {len(_ts_df):,} (30건 미만 제외)"
            # )

st.divider()

# 1. 헬퍼 함수 (투명도 처리)
def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        return f"rgba({r}, {g}, {b}, {alpha})"
    return f"rgba(100, 100, 100, {alpha})"

# 2. 레이더 차트 섹션 시작
st.subheader("브랜드별 속성 평가 - 긍정 vs 부정 입체 비교")

# 레이아웃 분할 (좌: 긍정, 우: 부정)
col_left, col_right = st.columns(2)

aspect_keys = [a["key"] for a in ASPECTS]
labels = [ASPECT_LABELS[k] for k in aspect_keys]
labels_closed = labels + [labels[0]]

# 공통 레이아웃 설정 함수
def update_radar_layout(fig, title, r_max: float = 1.0):
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=16)),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, r_max],
                tickformat=".0%",
                gridcolor="#ECEFF4"
            ),
            bgcolor="#F8FAFF"
        ),
        height=450,
        margin=dict(t=60, b=40, l=40, r=40),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )

# --- [왼쪽: 긍정(Positive) 레이더] ---
with col_left:
    fig_p = go.Figure()
    for brand in BRAND_ORDER:
        sub = polarity[polarity["brand"] == brand]
        if sub.empty: continue
        
        # P_ratio 데이터 추출
        vals = [float(sub[sub["aspect"] == asp]["P_ratio"].iloc[0]) if not sub[sub["aspect"] == asp].empty else 0.0 for asp in aspect_keys]
        vals_closed = vals + [vals[0]]
        
        color = BRANDS[brand]["color"]
        fig_p.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed,
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.2),
            line=dict(color=color, width=2),
            name=BRANDS[brand]["label"]
        ))
    
    update_radar_layout(fig_p, "긍정적 요인 (Positive)", r_max=1.0)
    st.plotly_chart(fig_p, use_container_width=True)

# --- [오른쪽: 부정(Negative) 레이더] ---
with col_right:
    fig_n = go.Figure()
    _n_vals_all = []
    for brand in BRAND_ORDER:
        sub = polarity[polarity["brand"] == brand]
        if sub.empty: continue

        vals = [float(sub[sub["aspect"] == asp]["N_ratio"].iloc[0]) if not sub[sub["aspect"] == asp].empty else 0.0 for asp in aspect_keys]
        _n_vals_all.extend(vals)
        vals_closed = vals + [vals[0]]

        color = BRANDS[brand]["color"]
        fig_n.add_trace(go.Scatterpolar(
            r=vals_closed, theta=labels_closed,
            fill="toself",
            fillcolor=hex_to_rgba(color, 0.2),
            line=dict(color=color, width=2),
            name=BRANDS[brand]["label"]
        ))

    # 실제 부정 비율 최댓값 기준으로 축 범위 자동 조정 (최소 10%, 최대 50%)
    _n_max = max(_n_vals_all) if _n_vals_all else 0.10
    _n_range = round(min(max(_n_max * 1.4, 0.10), 0.50), 2)
    update_radar_layout(fig_n, "부정적 요인 (Negative)", r_max=_n_range)
    st.plotly_chart(fig_n, use_container_width=True)

st.caption("면적이 넓을수록 해당 속성에서 감성 비율이 높음. 꼭짓점 호버로 수치 확인.")

st.divider()

# ── 브랜드 선택 → 발산 막대 ──────────────────────────────────
st.subheader("브랜드 강·약점 발산 분석")
sel_brand = st.radio(
    "브랜드 선택",
    options=[b for b in BRAND_ORDER if b in polarity["brand"].unique()],
    format_func=lambda b: BRANDS[b]["label"],
    horizontal=True,
)
with safe_block("발산 막대"):
    st.plotly_chart(aspect_polarity_diverging_bar(polarity, sel_brand), use_container_width=True)

st.divider()

# ── 강점·약점 Top3 카드 ───────────────────────────────────────
st.subheader("브랜드별 강점·약점 Top3")
active_brands = [b for b in BRAND_ORDER if b in polarity["brand"].unique()]
cols = st.columns(len(active_brands))
for col, brand in zip(cols, active_brands):
    sub   = polarity[polarity["brand"] == brand]
    top_p = sub.nlargest(3, "P_ratio")
    top_n = sub.nlargest(3, "N_ratio")
    color = BRANDS[brand]["color"]
    with col:
        st.markdown(
            f"<div style='border-top: 4px solid {color}; padding-top:8px;'>"
            f"<h4 style='margin:0;'>{BRANDS[brand]['label']}</h4></div>",
            unsafe_allow_html=True,
        )
        st.markdown("**강점**")
        for _, r in top_p.iterrows():
            st.markdown(f"- {ASPECT_LABELS[r['aspect']]} `{r['P_ratio']:.1%}`")
        st.markdown("**약점**")
        for _, r in top_n.iterrows():
            st.markdown(f"- {ASPECT_LABELS[r['aspect']]} `{r['N_ratio']:.1%}`")

st.divider()

# ── 대표 리뷰 예시 (속성 × 감성 단위) ────────────────────────
st.subheader("대표 리뷰 예시")
st.caption("속성·감성 조합별 실제 리뷰를 확인 — 모델 결과의 납득성 검증")

with safe_block("대표 리뷰"):
    rc1, rc2, rc3 = st.columns([1, 1, 1])
    with rc1:
        ex_brand = st.selectbox(
            "브랜드",
            options=[b for b in BRAND_ORDER if b in polarity["brand"].unique()],
            format_func=lambda b: BRANDS[b]["label"],
            key="ex_brand",
        )
    with rc2:
        ex_aspect = st.selectbox(
            "속성",
            options=[a["key"] for a in ASPECTS],
            format_func=lambda k: ASPECT_LABELS[k],
            key="ex_aspect",
        )
    with rc3:
        ex_polarity = st.radio(
            "감성", ["P", "N"], horizontal=True, key="ex_polarity",
            format_func=lambda x: "긍정 (P)" if x == "P" else "부정 (N)",
        )

    # ABSA 원본에서 해당 조합 매칭
    from utils.data_loader import get_absa, get_reviews
    try:
        absa = get_absa()
        reviews_for_ex = get_reviews(columns=("review_id", "brand", "rating", "content_clean"))

        if ex_aspect in absa.columns and not absa.empty:
            conf_col = f"{ex_aspect}_confidence"
            has_conf = conf_col in absa.columns
            pick_cols = ["review_id", conf_col] if has_conf else ["review_id"]
            
            ex_match = absa[absa[ex_aspect] == ex_polarity][pick_cols]
            sample = ex_match.merge(
                reviews_for_ex[reviews_for_ex["brand"] == ex_brand],
                on="review_id", how="inner",
            )
            
            # 내부적으로는 신뢰도나 평점으로 정렬하여 가장 좋은 예시를 뽑지만, 화면에는 굳이 설명하지 않음
            if has_conf:
                sample = sample.sort_values(conf_col, ascending=False)
            else:
                sample = sample.sort_values("rating", ascending=False)
                
            sample = sample.head(5)
            
            if sample.empty:
                empty_state("해당 조합의 리뷰가 없습니다.")
            else:
                st.caption(f"관련 리뷰 예시 {len(sample)}건") # 깔끔하게 건수만 표시
                
                for _, row in sample.iterrows():
                    rating = row.get("rating", "-")
                    text = str(row.get("content_clean", "")).strip()[:300]
                    polarity_color = "#002A54" if ex_polarity == "P" else "#C62828"
                    
                    st.markdown(
                        f"""<div style='border-left: 3px solid {polarity_color};
                        padding: 8px 12px; margin: 6px 0;
                        background: #FFFFFF; border: 1px solid #E5E5E5; border-radius: 8px;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.02);'>
                            <div style='font-size: 11px; color: #888; font-family: sans-serif;'>
                                별점 {rating}점
                            </div>
                            <div style='margin-top: 6px; color: #1A1A1A; line-height: 1.5;'>{text}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            empty_state(f"속성 '{ex_aspect}' 데이터를 찾을 수 없습니다.")
    except Exception as exc:
        empty_state("대표 리뷰 조회 실패", str(exc))

st.divider()

# ── [E] 브랜드별 부정 리뷰 핵심 키워드 카드 ──────────────────────
st.subheader("브랜드별 부정 리뷰 핵심 키워드")

sel_brand_neg = st.radio(
    "브랜드 선택",
    options=[b for b in BRAND_ORDER if b in polarity["brand"].unique()],
    format_func=lambda b: BRANDS[b]["label"],
    horizontal=True,
    key="sel_brand_neg_cards",
    label_visibility="collapsed"
)

if True:
    with safe_block("부정 리뷰 핵심 키워드 카드"):
        from utils.data_loader import get_absa as _get_absa, get_reviews as _get_reviews
        import collections
        import ast

        _absa_full = _get_absa()
        _reviews_all = _get_reviews(columns=("review_id", "brand", "tokens_topic"))
        _reviews_target = _reviews_all[_reviews_all["brand"] == sel_brand_neg]
        _neg_aspect_cols = [k for k in [a["key"] for a in ASPECTS] if k in _absa_full.columns]

        # 💡 1. 가장 진보된 불용어 (카테고리명, 결과적 행동, 1차원적 상태, 희귀 노이즈 전면 제거)
        global_stopwords = {
            '후기', '네이버페이', '작성', '리뷰', '제품', '상품', '구매', '브랜드', '매장', '안다르', '젝시믹스', '룰루레몬', '휠라', 'FILA', 'fila',
            '진짜', '너무', '그냥', '많이', '정말', '이거', '저거', '정도', '조금', '약간', '생각', '마음', '느낌', '부분', '때문', '처음',
            '하다', '있다', '없다', '같다', '그렇다', '나다', '보다', '가다', '오다', '다르다', '많다', '들다', '입다', '신다',
            '좋다', '예쁘다', '편하다', '만족', '최고', '잘맞다', '이쁘다', '가볍다', '부드럽다', '괜찮다', '어울리다', '맞다', '마음에들다', '편안',
            '심하다', '생기다', '잡다', '이상', '귀찮다', '부담', '보내다', '확인', '주다', '맞추다', '느껴지다',
            '반품', '교환', '환불', '실망', '아쉽다', '불편', '별로', '최악', '고민', '기다리다', '시간', '택배', '대비',
            '사이즈', '신발', '운동화', '레깅스', '바지', '디자인', '색상', '컬러', '가격', '소재', '재질', '착용감', '착용', '원단',
            '작다', '크다', '길다', '짧다', '넓다', '좁다', 
            '여름', '겨울', '사진', '실물', '정사이즈', '치수', '운동', '요가', '러닝', '하루', '연락', '핸드폰', '버전', '미니', '사각', '삭스', '원피스'
        }

        aspect_buckets = {}
        global_counter = collections.Counter()
        word_scores_all = collections.defaultdict(dict)

        # 💡 2. 빈도 집계 (전역 불용어 적용)
        for _asp in _neg_aspect_cols:
            _neg_match = _absa_full[(_absa_full[_asp] == "N") & (_absa_full["review_id"].isin(_reviews_target["review_id"]))]
            _neg_ids = _neg_match["review_id"]
            
            _neg_tokens_series = _reviews_target[_reviews_target["review_id"].isin(_neg_ids)]["tokens_topic"].dropna()
            
            _words = []
            for val in _neg_tokens_series:
                if isinstance(val, str):
                    if val.startswith('['):
                        try: tokens = ast.literal_eval(val)
                        except: tokens = []
                    else: tokens = val.split()
                elif isinstance(val, list): tokens = val
                else: tokens = []
                _words.extend(tokens)
            
            filtered_words = [w for w in _words if len(w) > 1 and not w.isdigit() and w not in global_stopwords]
            
            local_counter = collections.Counter(filtered_words)
            aspect_buckets[_asp] = {
                "label": ASPECT_LABELS[_asp],
                "counter": local_counter,
                "count": len(_neg_ids)
            }
            global_counter.update(filtered_words)

        # 💡 3. TF-ICF 알고리즘을 통한 '단어별 최적 속성 1:1 독점 할당'
        exclusive_buckets = collections.defaultdict(dict)
        
        for _asp in _neg_aspect_cols:
            if _asp not in aspect_buckets: continue
            local_counter = aspect_buckets[_asp]["counter"]
            
            for w, local_freq in local_counter.items():
                if local_freq >= 2: # 최소 2회 이상 등장한 유효 단어만
                    global_freq = global_counter[w]
                    score = (local_freq ** 2) / global_freq
                    word_scores_all[w][_asp] = score
        
        # 각 단어를 가장 가중치가 높은 단 하나의 속성에만 배정
        for w, scores in word_scores_all.items():
            best_asp = max(scores, key=scores.get)
            exclusive_buckets[best_asp][w] = scores[best_asp]

        # 💡 4. 렌더링 (독점 할당된 명단에서만 Top3 추출)
        _neg_cards = []
        for _asp in _neg_aspect_cols:
            bucket = aspect_buckets.get(_asp)
            if not bucket or bucket["count"] == 0:
                continue
                
            scored_words = exclusive_buckets.get(_asp, {})
            _top3 = [w for w, score in sorted(scored_words.items(), key=lambda x: x[1], reverse=True)[:3]]
            
            _neg_cards.append((bucket["label"], _top3, bucket["count"]))

        if _neg_cards:
            _nc_cols = st.columns(len(_neg_cards))
            for _col, (_asp_label, _kws, _cnt) in zip(_nc_cols, _neg_cards):
                with _col:
                    st.markdown(
                        f"<div style='border-top:3px solid #C62828; padding:12px 14px;"
                        f"background:#FFFFFF; border: 1px solid #E5E5E5; border-radius:8px;"
                        f"box-shadow: 0 1px 2px rgba(0,0,0,0.02); margin-bottom:10px;'>"
                        f"<div style='font-size:11px; color:#C62828; font-weight:600; font-family:sans-serif;'>"
                        f"부정 — {_asp_label} ({_cnt:,}건)</div>"
                        f"<div style='font-size:15px; font-weight:700; margin-top:8px; color:#1A1A1A;'>"
                        f"{' · '.join(_kws) if _kws else '—'}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption(f"{BRANDS[sel_brand_neg]['label']}의 부정 리뷰 데이터가 없습니다.")

st.divider()

# ── 원시 데이터 ──────────────────────────────────────────────
with st.expander("ABSA 집계 원시 데이터"):
    df_view = polarity.copy()
    df_view["aspect"] = df_view["aspect"].map(ASPECT_LABELS)
    for c in ["P_ratio", "N_ratio", "X_ratio"]:
        df_view[c] = (df_view[c] * 100).round(1).astype(str) + "%"
    st.dataframe(df_view, use_container_width=True)
