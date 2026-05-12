"""
positioning_map.py — 핵심 시각화: 동적 포지셔닝 맵 (2D 산점도)
================================================================

기능:
- 4개 브랜드를 (x_function, y_heritage) 좌표에 배치.
- 마우스 호버 시 핵심 토픽, 강점/약점 ABSA 점수, 리뷰 수, 평점을 툴팁으로 노출.
- 95% 신뢰구간을 ellipse(또는 bar)로 표시 → 통계적 유의성 시각화.
- 사분면 라벨 (전략적 포지셔닝 영역) 자동 표시.
- 휠라(자사)는 별 모양 마커 + 점선 화살표로 권장 이동 방향 시각화.

사분면 의미:
    II. Heritage Premium      |  I. Holistic Leader (Lululemon 가설)
    --------------------------+--------------------------
    III. Mass Functional      |  IV. Function-First Performer
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from config import BRANDS, BRAND_COLORS, BRAND_ORDER, POSITIONING_AXIS_RANGE


def render_positioning_map(
    pos_df: pd.DataFrame,
    polarity_df: pd.DataFrame | None = None,
    topic_meta: pd.DataFrame | None = None,
    show_ci: bool = True,
    show_quadrants: bool = True,
    target_position: tuple[float, float] | None = None,
    height: int = 640,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    x_title: str = "기능성 (Functionality) →",
    y_title: str = "브랜드 헤리티지 (Heritage) ↑",
) -> go.Figure:
    """포지셔닝 맵 메인 렌더러.

    Args:
        pos_df: 브랜드 좌표 (compute_positioning_from_absa 결과)
        polarity_df: ABSA 속성 비율 (호버용)
        topic_meta: 토픽 메타 (호버용 — 브랜드별 top topic 추출은 별도)
        show_ci: 신뢰구간 막대 표시
        show_quadrants: 사분면 분할선 표시
        target_position: 휠라 권장 위치 (예: (0.7, 0.7))
        x_range / y_range: 축 범위 (None=기본 POSITIONING_AXIS_RANGE)
        x_title / y_title: 축 라벨 — 산식 전환 시 의미 명확화용
    """
    _x_range = x_range if x_range is not None else POSITIONING_AXIS_RANGE
    _y_range = y_range if y_range is not None else POSITIONING_AXIS_RANGE
    fig = go.Figure()

    if pos_df.empty:
        return _empty_map()

    # ── 1. 사분면 분할선 + 라벨 ────────────────────────────
    if show_quadrants:
        _add_quadrants(fig, _x_range, _y_range)

    # ── 2. 브랜드별 신뢰구간 → 산점 ─────────────────────────
    skipped_brands: list[str] = []
    for _, row in pos_df.iterrows():
        brand = row["brand"]
        if brand not in BRANDS:
            continue
        meta = BRANDS[brand]
        # 좌표 결측치는 좌하단으로 그리지 않고 스킵 (전략 오해 방지)
        if pd.isna(row["x_function"]) or pd.isna(row["y_heritage"]):
            skipped_brands.append(meta["label"])
            continue
        x, y = float(row["x_function"]), float(row["y_heritage"])

        # 신뢰구간 (가로/세로 막대)
        if show_ci and "x_function_ci_low" in row.index:
            fig.add_shape(
                type="line",
                x0=row["x_function_ci_low"], x1=row["x_function_ci_high"],
                y0=y, y1=y,
                line=dict(color=meta["color"], width=1, dash="dot"),
                opacity=0.5,
            )
            fig.add_shape(
                type="line",
                x0=x, x1=x,
                y0=row["y_heritage_ci_low"], y1=row["y_heritage_ci_high"],
                line=dict(color=meta["color"], width=1, dash="dot"),
                opacity=0.5,
            )

        # 호버 텍스트 빌드
        hover = _build_hover_text(brand, row, polarity_df, topic_meta)

        # 자사는 별 모양, 경쟁사는 원
        marker_symbol = "star" if meta["is_self"] else "circle"
        marker_size = 32 if meta["is_self"] else 22

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(
                symbol=marker_symbol,
                size=marker_size,
                color=meta["color"],
                line=dict(width=2, color="white"),
            ),
            text=[meta["label"]],
            textposition="top center",
            textfont=dict(size=13, color="white"),
            name=meta["label"],
            customdata=[hover],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=True,
        ))

    # ── 3. 휠라 권장 이동 화살표 ────────────────────────────
    if target_position is not None:
        fila_row = pos_df[pos_df["brand"] == "FILA"]
        if not fila_row.empty and pd.notna(fila_row["x_function"].iloc[0]) and pd.notna(fila_row["y_heritage"].iloc[0]):
            fx, fy = float(fila_row["x_function"].iloc[0]), float(fila_row["y_heritage"].iloc[0])
            tx, ty = target_position
            fig.add_annotation(
                x=tx, y=ty, ax=fx, ay=fy, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.4, arrowwidth=2,
                arrowcolor="#FFD700",
            )
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty], mode="markers+text",
                marker=dict(symbol="x", size=18, color="#FFD700"),
                text=["권장"], textposition="bottom center",
                textfont=dict(size=11, color="#FFD700"),
                name="권장 위치", showlegend=True,
                hovertemplate="<b>휠라 권장 포지셔닝</b><br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
            ))

    # ── 3.5. 산출 불가 브랜드 안내 ────────────────────────────
    if skipped_brands:
        fig.add_annotation(
            text=f"⚠ 산출 불가 (NA): {', '.join(skipped_brands)}",
            xref="paper", yref="paper", x=0.99, y=1.06,
            showarrow=False, font=dict(size=11, color="#FF9800"),
            xanchor="right",
        )

    # ── 4. 레이아웃 ────────────────────────────────────────
    fig.update_layout(
        title="브랜드 포지셔닝 맵 — 기능성 × 헤리티지",
        xaxis=dict(
            title=x_title,
            range=list(_x_range),
            zeroline=False, gridcolor="#33334d",
            tickformat=".2f",
        ),
        yaxis=dict(
            title=y_title,
            range=list(_y_range),
            zeroline=False, gridcolor="#33334d",
            tickformat=".2f",
        ),
        height=height,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"),
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.15,
            xanchor="center", x=0.5,
        ),
    )
    return fig


def _add_quadrants(fig: go.Figure,
                   x_range: tuple[float, float] = (0.0, 1.0),
                   y_range: tuple[float, float] = (0.0, 1.0)) -> None:
    """사분면 가이드라인 — 축 범위에 맞춰 분할 중심·라벨 위치 동적 계산."""
    xlo, xhi = x_range
    ylo, yhi = y_range
    xmid = (xlo + xhi) / 2
    ymid = (ylo + yhi) / 2
    fig.add_shape(type="line", x0=xmid, x1=xmid, y0=ylo, y1=yhi,
                  line=dict(color="#555", width=1, dash="dash"))
    fig.add_shape(type="line", x0=xlo, x1=xhi, y0=ymid, y1=ymid,
                  line=dict(color="#555", width=1, dash="dash"))

    qx_lo = xlo + (xhi - xlo) * 0.25
    qx_hi = xlo + (xhi - xlo) * 0.75
    qy_lo = ylo + (yhi - ylo) * 0.05
    qy_hi = ylo + (yhi - ylo) * 0.95

    annotations = [
        (qx_lo, qy_hi, "II. Heritage Premium", "#999"),
        (qx_hi, qy_hi, "I. Holistic Leader",   "#FFD700"),
        (qx_lo, qy_lo, "III. Mass Functional", "#999"),
        (qx_hi, qy_lo, "IV. Function-First",   "#999"),
    ]
    for x, y, text, color in annotations:
        fig.add_annotation(
            x=x, y=y, text=f"<i>{text}</i>",
            showarrow=False, font=dict(size=11, color=color),
            bgcolor="rgba(14,17,23,0.7)", borderpad=4,
        )


def _build_hover_text(brand: str, row: pd.Series,
                      polarity_df: pd.DataFrame | None,
                      topic_meta: pd.DataFrame | None) -> str:
    """호버 툴팁 HTML — Plotly customdata 로 전달."""
    meta = BRANDS[brand]
    n_reviews = int(row.get("n_reviews", 0)) if pd.notna(row.get("n_reviews", None)) else 0
    mean_rating = row.get("mean_rating", float("nan"))

    parts = [
        f"<b style='font-size:14px; color:{meta['color']}'>{meta['label']}</b>",
        f"<br> 리뷰 <b>{n_reviews:,}</b>건",
        f"<br> 평점 <b>{mean_rating:.2f}</b>" if pd.notna(mean_rating) else "",
        f"<br> 좌표 ({row['x_function']:.2f}, {row['y_heritage']:.2f})",
    ]

    # ABSA 강/약점 (polarity_df 가 있으면)
    if polarity_df is not None and not polarity_df.empty:
        sub = polarity_df[polarity_df["brand"] == brand].copy()
        if not sub.empty:
            from config import ASPECT_LABELS
            sub["aspect_label"] = sub["aspect"].map(ASPECT_LABELS)
            top_p = sub.nlargest(2, "P_ratio")[["aspect_label", "P_ratio"]]
            top_n = sub.nlargest(2, "N_ratio")[["aspect_label", "N_ratio"]]
            parts.append("<br><br><b> 강점</b>")
            for _, r in top_p.iterrows():
                parts.append(f"<br> · {r['aspect_label']} {r['P_ratio']:.1%}")
            parts.append("<br><b> 약점</b>")
            for _, r in top_n.iterrows():
                parts.append(f"<br> · {r['aspect_label']} {r['N_ratio']:.1%}")

    # 대표 토픽
    if topic_meta is not None and "top_topics" in row.index:
        topics = row.get("top_topics") or []
        if isinstance(topics, list) and topics:
            parts.append(f"<br><br> <b>핵심 토픽</b><br> · " + "<br> · ".join(topics[:3]))

    return "".join(parts)


def _empty_map() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="포지셔닝 산출 데이터 없음<br>모델팀 ABSA 결과 도착 후 자동 표시",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=15, color="gray"),
    )
    fig.update_layout(height=600, xaxis_visible=False, yaxis_visible=False)
    return fig
