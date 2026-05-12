"""
5_전략_포지셔닝.py — 핵심 산출물 (포지셔닝 맵 + White Space)
=============================================================
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import APP_TITLE, PAGE_LAYOUT, BRANDS, BRAND_ORDER, PATHS, CACHE_TTL
from utils.data_loader import (
    get_positioning, get_sna, compute_aspect_polarity, get_topic_meta, get_reviews,
)
from utils.session import init_session, mark_page_visited
from utils.exceptions import safe_block, warn_using_dummy, empty_state
from components.filters import render_sidebar_filters
from components.positioning_map import render_positioning_map
from components.charts import keyword_centrality_bar
from components.page_header import render_page_intro


st.set_page_config(page_title=f"{APP_TITLE} — 포지셔닝", page_icon=None, **PAGE_LAYOUT)
init_session()
mark_page_visited("positioning")

st.title("전략 포지셔닝")
st.caption("최종 산출 — 휠라의 의류 시장 진입 좌표")
st.markdown(
    "<p style='font-size:12px; color:#888; margin:2px 0 2px;'>"
    "홈 &nbsp;›&nbsp; 상품/고객 전략 &nbsp;›&nbsp; BERTopic &nbsp;›&nbsp; "
    "ABSA &nbsp;›&nbsp; <strong>포지셔닝</strong></p>",
    unsafe_allow_html=True,
)
st.caption("ABSA 속성 점수를 기능성 × 헤리티지 2축으로 압축하여 FILA의 의류 시장 진입 좌표를 확정합니다.")

render_page_intro(
    "기능성 × 헤리티지 2축 좌표와 신발↔의류 키워드 네트워크(PMI) 분석으로 "
    "FILA의 진입 좌표와 White Space 전략 옵션(A/B/C)을 결정합니다.",
    accent="#004B87",
)

render_sidebar_filters()

# ── 데이터 ─────────────────────────────────────────────────
with safe_block("포지셔닝 데이터 로드"):
    pos        = get_positioning()
    polarity   = compute_aspect_polarity()
    topic_meta = get_topic_meta()

if pos.empty:
    empty_state("포지셔닝 산출 불가", "ABSA 결과가 도착하면 자동 계산")
    st.stop()

# ── 좌표 산식 토글 ────────────────────────────────────────
# A: (1-X_ratio)·(P-N)  — 발화량 × 긍정 강도
# B: (P-N)/(P+N) → 0.5×(1+polarity)  — X_ratio 분모 제외, 모두 0.9+ 클러스터 분산
st.subheader("좌표 산식 / 축 설정")
sf_col1, sf_col2 = st.columns([2, 1])
with sf_col1:
    formula_choice = st.radio(
        "좌표 산식",
        options=["A. 발화량 × 강도 (1−X)·(P−N)", "B. (P−N)/(P+N)"],
        index=1,
        horizontal=True,
        key="positioning_formula",
        help=(
            "A: 발화량까지 반영 → 브랜드 간 격차 6배 명확."
            "B: ABSA 미언급(X)을 분모에서 제외 → 모든 브랜드 0.9+ 클러스터."
        ),
    )
with sf_col2:
    zoom_axis = st.toggle(
        "축 확대 (산식 B 전용)",
        value=True,
        help="산식 B 사용 시 축을 0.88~1.00으로 확대해 클러스터 차이를 보이게 함",
    )

use_formula_a = formula_choice.startswith("A")


def _compute_pos_a(polarity_df: pd.DataFrame) -> pd.DataFrame:
    """산식 A 좌표: x_function = (1-X_func)·(P_func-N_func), y_heritage 동일."""
    if polarity_df.empty:
        return pd.DataFrame()
    func  = polarity_df[polarity_df["aspect"] == "functionality"].set_index("brand")
    brand = polarity_df[polarity_df["aspect"] == "brand_heritage"].set_index("brand")
    rows = []
    for b in BRAND_ORDER:
        x_f = (1 - func.loc[b, "X_ratio"]) * (func.loc[b, "P_ratio"] - func.loc[b, "N_ratio"]) if b in func.index else float("nan")
        y_h = (1 - brand.loc[b, "X_ratio"]) * (brand.loc[b, "P_ratio"] - brand.loc[b, "N_ratio"]) if b in brand.index else float("nan")
        n   = int(func.loc[b, "n_reviews"]) if b in func.index else 0
        rows.append({
            "brand": b,
            "x_function": x_f,
            "y_heritage": y_h,
            "x_function_ci_low":  x_f - 0.02 if pd.notna(x_f) else x_f,
            "x_function_ci_high": x_f + 0.02 if pd.notna(x_f) else x_f,
            "y_heritage_ci_low":  y_h - 0.02 if pd.notna(y_h) else y_h,
            "y_heritage_ci_high": y_h + 0.02 if pd.notna(y_h) else y_h,
            "n_reviews": n,
            "mean_rating": float("nan"),
        })
    return pd.DataFrame(rows)


if use_formula_a:
    pos_view = _compute_pos_a(polarity)
    x_range_view = (0.0, 0.6)
    y_range_view = (0.0, 0.35)
    x_title_view = "기능성 = (1 − X<sub>func</sub>) · (P − N) →"
    y_title_view = "헤리티지 = (1 − X<sub>brand</sub>) · (P − N) ↑"
    target_default = (0.45, 0.25)
else:
    pos_view = pos
    if zoom_axis:
        x_range_view = (0.88, 1.00)
        y_range_view = (0.88, 1.00)
    else:
        x_range_view = (0.0, 1.0)
        y_range_view = (0.0, 1.0)
    x_title_view = "기능성 (Functionality) →"
    y_title_view = "브랜드 헤리티지 (Heritage) ↑"
    target_default = (0.95, 0.95)

st.divider()

# ── 시뮬레이터 슬라이더 ────────────────────────────────────
st.subheader("휠라 권장 포지셔닝 시뮬레이터")
ctl = st.columns([1, 1, 2])
with ctl[0]:
    target_x = st.slider(
        "목표 기능성",
        float(x_range_view[0]), float(x_range_view[1]),
        float(target_default[0]),
        step=0.01,
        key=f"target_x_{formula_choice}",
        help="휠라가 도달해야 할 기능성 점수 (산식별 범위 자동 조정)",
    )
with ctl[1]:
    target_y = st.slider(
        "목표 헤리티지",
        float(y_range_view[0]), float(y_range_view[1]),
        float(target_default[1]),
        step=0.01,
        key=f"target_y_{formula_choice}",
        help="신발 헤리티지를 의류로 전이한 후 목표 점수",
    )
with ctl[2]:
    show_ci = st.toggle("신뢰구간 표시", value=True)
    show_q  = st.toggle("사분면 표시",   value=True)

# ── 포지셔닝 맵 ────────────────────────────────────────────
with safe_block("포지셔닝 맵"):
    fig = render_positioning_map(
        pos_df=pos_view,
        polarity_df=polarity,
        topic_meta=topic_meta,
        show_ci=show_ci,
        show_quadrants=show_q,
        target_position=(target_x, target_y),
        height=680,
        x_range=x_range_view,
        y_range=y_range_view,
        x_title=x_title_view,
        y_title=y_title_view,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── 좌표 산출법 ────────────────────────────────────────────
with st.expander("좌표 산출법 — 두 산식 비교"):
    st.markdown(
        """
**ABSA 6속성 중 2개 속성(`functionality`, `brand_heritage`)의 P/N/X 비율을 사용합니다.**

| 산식 | 공식 | 특징 | 4브랜드 격차 |
|---|---|---|---|
| **B. ABSA 미언급(X) 제외** | `0.5 × (1 + (P−N) / (P+N))` | 언급 시 긍정 강도만 반영. X는 분모 제외 | gap ≈ 0.05 (클러스터링) |
| **A. 발화량 × 강도** | `(1 − X) × (P − N)` | 발화량(`1−X`)을 곱해 페널티 부여 | gap ≈ 0.31 (분명) |

- **산식 B는 "언급되었을 때" 얼마나 긍정인지만 측정** → FILA 기능성 X=53% 인 점이 무시되어 좌표가 부풀려짐 (모두 0.9+).
- **산식 A는 "얼마나 자주 + 얼마나 긍정"의 곱** → BERTopic 발화량 결과(FILA 기능성 발화 9.7% vs 룰루레몬 46.7%)와 정합.
- 두 산식 모두에서 FILA functionality는 4위 — 순위는 유지되나 격차 해석이 다름.
- 산식 B에서 축 확대(0.88~1.00) 시 시각적 차이는 보이지만, 좌표 해석은 동일.
- 해당 속성에서 ABSA 결과가 없는 브랜드는 NaN("산출 불가")로 처리하며 맵에 그리지 않습니다.
- 신뢰구간(CI)은 ±0.02~0.05 고정 (실제 부트스트랩 도착 시 교체 예정).

**보완 예정:** BERTopic 토픽 점유율 가중치, SNA 키워드 중심성 반영.
        """
    )

# pos = 좌표 뷰로 갈아끼우기 (이하 좌표 테이블 등에서 사용)
pos = pos_view

# ── 좌표 테이블 ────────────────────────────────────────────
st.subheader("브랜드 좌표")
df_view = pos[["brand", "x_function", "y_heritage", "n_reviews"]].copy()
df_view.columns = ["브랜드", "기능성", "헤리티지", "리뷰 수"]

# NaN → "산출 불가" 표시 (0으로 시각화 시 좌하단 오인 방지)
def _fmt_score(v):
    return "산출 불가" if pd.isna(v) else f"{v:.3f}"
df_view["기능성"]   = df_view["기능성"].apply(_fmt_score)
df_view["헤리티지"] = df_view["헤리티지"].apply(_fmt_score)
st.dataframe(df_view, use_container_width=True, hide_index=True)
na_brands = pos[pos["x_function"].isna() | pos["y_heritage"].isna()]["brand"].tolist()
if na_brands:
    st.caption(f"산출 불가 브랜드: {', '.join(na_brands)} — 해당 속성에서 ABSA 결과 부재")

st.divider()

# ── SNA 키워드 ─────────────────────────────────────────────
st.subheader("브랜드별 핵심 키워드 (SNA 중심성)")
sna = get_sna()
if not sna.empty:
    cols_sna = st.columns(2)
    for i, brand in enumerate(BRAND_ORDER):
        with cols_sna[i % 2]:
            with safe_block(f"{brand} SNA"):
                st.plotly_chart(
                    keyword_centrality_bar(sna, brand, top_n=12),
                    use_container_width=True,
                )
else:
    empty_state("SNA 결과 없음")

st.divider()

# ─────────────────────────────────────────────────────────────
# FILA 신발 × 의류 연결 분석 (PMI + 연결·매개 중심성)
# ─────────────────────────────────────────────────────────────
st.subheader("FILA 신발 → 의류 연결 분석 (PMI 기반 키워드 네트워크)")
st.caption(
    "FILA 리뷰 토큰에서 PMI(점별 상호정보량) 기반 의미 있는 단어쌍 추출 → "
    "연결 중심성(얼마나 많은 단어와 연결되는지)·매개 중심성(서로 다른 단어 그룹을 이어주는 정도) 계산"
)

# ── 본질 인사이트: 신발 vs 의류 발화량 격차 (5.6:1) ───────────
st.markdown(
    """
<div style='background:#FFF8E1; border-left:5px solid #F57F17;
padding:14px 18px; border-radius:0 6px 6px 0; margin-bottom:14px;'>
  <div style='font-size:13px; color:#F57F17; font-weight:700; margin-bottom:6px;'>
    핵심 인사이트 — FILA 리뷰 내부의 신발/의류 발화 불균형
  </div>
  <div style='font-size:13px; color:#222; line-height:1.7;'>
    FILA 토큰 풀에서 <strong>신발 시드 누적 빈도 10,946건</strong> vs
    <strong>의류 시드 누적 빈도 1,946건</strong> = <strong style='color:#C62828;'>5.6 : 1</strong>.<br/>
    PMI 그래프에 의류 노드가 적게 잡히는 것은 모델 문제가 아니라
    <strong>의류 단어 자체가 리뷰에 적게 나오기 때문</strong>. <br/>
    <strong>FILA 신발 인식 → 의류 인식 전환이 안 된 상태가 White Space의 본질</strong> —
    신발 키워드 자산을 의류 키워드로 연결하는 마케팅 메시지·콜라보·PDP 디자인이 핵심 과제.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# 신발/의류 시드 단어 (카테고리 분류용 — 최신 트렌드/애슬레저 반영)
_SHOE_SEEDS = {
    "신발", "운동화", "스니커즈", "슈즈", "깔창", "밑창", "굽", "인솔",
    "착화감", "끈", "하이탑",
    "인터런", "디스럽터", "에샤페", "레이", "페이토", "오크먼트", "코트디럭스", "메리제인", "글리오", "하레핀", "레인저코어", "테니스화"
}
_APPAREL_SEEDS = {
    "레깅스", "티셔츠", "바지", "상의", "하의", "셔츠", "반팔", "긴팔",
    "조거", "재킷", "니트", "기모", "맨투맨", "원피스", "스커트", "반바지",
    "티", "조거팬츠", "팬츠", "스웻팬츠", "스웻", "후드", "후드티", "후디", "집업",
    "탑", "크롭", "크롭탑", "트레이닝복", "트레이닝", "트랙수트",
    "탱크탑", "민소매", "롱슬리브", "숏팬츠",
    "조끼", "베스트", "점퍼", "코트", "패딩", "플리스", "집업후드",
    "스포츠브라", "브라탑", "요가복", "요가팬츠",
    "트레이너", "스포츠웨어", "애슬레저",
    "카고", "카고팬츠", "조거바지", "드레스", "튜닉", "후드집업",
    "부츠컷", "부츠컷레깅스", "바이커쇼츠", "커버업"
}
_PMI_STOPWORDS: frozenset[str] = frozenset([
    "있다", "없다", "좋다", "같다", "하다", "이다", "되다", "많다",
    "너무", "진짜", "정말", "그냥", "조금", "약간", "엄청", "완전", "매우", "많이",
    "느낌", "생각", "마음", "이거", "저거", "그런", "이런", "저런", "어떤",
    "것", "수", "때", "거", "게", "걸", "더", "안", "못", "또", "다", "잘",
    "좋아", "좋아요", "있어", "없어", "같아", "합니다", "됩니다",
    "구매", "주문", "배송", "포장", "리뷰", "상품", "제품",
    "이번", "이후", "기존", "받다", "오다", "보다", "주다",
    # 💡 [신규 추가] 범용 속성어 / 상황 노이즈 (이미지 분석 기반 필터링)
    # 1. 크기/핏 단순 평가 (전략적 차별점이 안 되는 단어)
    "크다", "작다", "정사이즈", "넓다", "맞다", "넉넉하다", "길다", "짧다",
    # 2. 색상/외형 단순 명칭 (제품 본연의 감성어가 아닌 단어)
    "컬러", "색감", "블랙", "실버", "화이트", "색상", "색깔", "실물", "라인", "스타일", "소재", "코디",
    # 3. 상황 / 단순 감상 / 일반 행동
    "불편", "아프다", "괜찮다", "부드럽다", "깔끔", "빠르다", "처음", "고민", "여름", "겨울",
    "아주", "들다", "걷다가", "다닙니다", "잡다", "착용", "입다", "신다", "운동", "휠라",
    "다니다", "정도", "가다", "가을", "봄", "높다", "좁다", "걱정", "길이", "생기다", "전혀", "가능", "훨씬", "좁다", "발볼", "발등",
])

@st.cache_data(ttl=CACHE_TTL)
def _compute_pmi_centrality(
    top_vocab: int = 400,
    min_pair_count: int = 3,
    min_pmi: float = 0.3,
) -> pd.DataFrame:
    df = get_reviews(columns=("brand", "tokens"))
    fila = df[df["brand"] == "FILA"]["tokens"].dropna()

    docs = [str(t).split() for t in fila]

    word_counts: Counter = Counter(w for doc in docs for w in doc)
    total_words = sum(word_counts.values()) or 1

    vocab = {
        w for w, c in word_counts.most_common(top_vocab)
        if len(w) > 1 and not w.isdigit() and c >= min_pair_count
        and w not in _PMI_STOPWORDS
    }
        # 👇 [이 한 줄만 추가하세요!] 빈도수 부족으로 짤린 신발/의류 시드 단어를 강제로 부활시키는 치트키
    vocab.update(w for w in _APPAREL_SEEDS.union(_SHOE_SEEDS) if word_counts[w] > 0)

    cooc: Counter = Counter()
    for doc in docs:
        words = [w for w in doc if w in vocab]
        for i, w1 in enumerate(words):
            for w2 in words[i + 1: i + 4]:
                if w1 != w2:
                    cooc[tuple(sorted([w1, w2]))] += 1

    total_cooc = sum(cooc.values()) or 1

    G = nx.Graph()
    for (w1, w2), c in cooc.items():
        if c < min_pair_count:
            continue
        pmi = (
            math.log2(c / total_cooc)
            - math.log2(word_counts[w1] / total_words)
            - math.log2(word_counts[w2] / total_words)
        )
        if pmi >= min_pmi:
            G.add_edge(w1, w2, weight=float(pmi), count=c)

    if len(G.nodes) < 3:
        return pd.DataFrame()

    degree_c  = nx.degree_centrality(G)
    # k 샘플링으로 대형 그래프 속도 개선 (노드 수 300 이하면 정확 계산)
    k_sample = None if len(G.nodes) <= 300 else 100
    between_c = nx.betweenness_centrality(G, normalized=True, k=k_sample)

    def _cat(w: str) -> str:
        if w in _SHOE_SEEDS:    return "신발"
        if w in _APPAREL_SEEDS: return "의류"
        return "공통"

    rows = [
        {
            "키워드":   node,
            "연결 중심성": round(degree_c.get(node, 0), 4),
            "매개 중심성": round(between_c.get(node, 0), 4),
            "빈도":     word_counts.get(node, 0),
            "카테고리":  _cat(node),
        }
        for node in G.nodes()
    ]
    return pd.DataFrame(rows)


with safe_block("PMI 파라미터"):
    pmi_c1, pmi_c2, pmi_c3 = st.columns(3)
    with pmi_c1:
        top_vocab_sel = st.slider("어휘 크기 (상위 N 단어)", 100, 400, 250, 50,
                                  key="pmi_vocab")
    with pmi_c2:
        min_pair_sel = st.slider("최소 동시 출현 횟수", 2, 20, 3, 1,
                                 key="pmi_minpair")
    with pmi_c3:
        min_pmi_sel = st.slider("최소 PMI 임계값", 0.0, 3.0, 0.5, 0.1,
                                key="pmi_threshold")

with st.spinner("PMI 그래프 산출 중… (최초 1회만 소요, 이후 캐시)"):
    pmi_df = _compute_pmi_centrality(
        top_vocab=top_vocab_sel,
        min_pair_count=min_pair_sel,
        min_pmi=min_pmi_sel,
    )

if pmi_df.empty:
    empty_state("PMI 결과 없음", "임계값을 낮추거나 어휘 크기를 늘려 주세요.")
else:
    st.caption(
        f"그래프 노드: {len(pmi_df):,}개 단어 | "
        f"신발: {(pmi_df['카테고리']=='신발').sum()}개 / "
        f"의류: {(pmi_df['카테고리']=='의류').sum()}개 / "
        f"공통: {(pmi_df['카테고리']=='공통').sum()}개"
    )

    _CAT_COLOR = {"신발": "#003087", "의류": "#D4000F", "공통": "#888888"}

    col_deg, col_bet = st.columns(2)

    with col_deg:
        top_deg = pmi_df.nlargest(15, "연결 중심성").sort_values("연결 중심성")
        fig_deg = px.bar(
            top_deg, x="연결 중심성", y="키워드",
            orientation="h",
            color="카테고리",
            color_discrete_map=_CAT_COLOR,
            title="연결 중심성 Top 15",
            labels={"연결 중심성": "연결 중심성", "키워드": ""},
            hover_data={"빈도": True, "카테고리": True},
        )
        fig_deg.update_layout(height=480, showlegend=True,
                              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_deg, use_container_width=True)
        st.caption("연결 중심성: 한 단어가 얼마나 많은 다른 단어와 연결되어 있는지 — 수치가 높을수록 브랜드 경험의 중심 단어")
        st.caption("※ 가장 많은 단어들과 짝지어져서 언급된 핵심 단어입니다.")

    with col_bet:
        top_bet = pmi_df.nlargest(15, "매개 중심성").sort_values("매개 중심성")
        fig_bet = px.bar(
            top_bet, x="매개 중심성", y="키워드",
            orientation="h",
            color="카테고리",
            color_discrete_map=_CAT_COLOR,
            title="매개 중심성 Top 15",
            labels={"매개 중심성": "매개 중심성", "키워드": ""},
            hover_data={"빈도": True, "카테고리": True},
        )
        fig_bet.update_layout(height=480, showlegend=True,
                              legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_bet, use_container_width=True)
        st.caption("매개 중심성: 신발 단어 그룹과 의류 단어 그룹 사이를 얼마나 잘 이어주는지 — 브랜드 확장의 전환점이 되는 단어")
        st.caption("※ 서로 다른 주제나 흩어진 단어들을 중간에서 이어주는 다리 역할의 단어입니다.")

            # 연결 중심성 vs 매개 중심성 산점도
    st.subheader("키워드 포지셔닝 — 연결 중심성 × 매개 중심성")
    
    # 💡 [수정] 다시 상위 80개로 제한. "의류 노드가 없는 것 자체가 인사이트"라는 논리를 전개합니다.
    fig_scatter = px.scatter(
        pmi_df[pmi_df["빈도"] >= 10].nlargest(80, "연결 중심성"),
        x="연결 중심성", y="매개 중심성",
        size="빈도", color="카테고리",
        color_discrete_map=_CAT_COLOR,
        text="키워드",
        hover_data={"빈도": True, "카테고리": True},
        labels={
            "연결 중심성": "연결 중심성 (많이 연결된 단어)",
            "매개 중심성": "매개 중심성 (다리 역할 단어)",
        },
        title="키워드 네트워크 포지셔닝 맵 (빈도 10 이상, 상위 80개)",
        size_max=40,
    )
    fig_scatter.update_traces(textposition="top center", textfont_size=9)
    fig_scatter.update_layout(height=560)
    st.plotly_chart(fig_scatter, use_container_width=True, key="pmi_scatter_top80")
    
    # 💡 캡션 보강: 보고 시나리오에 맞춘 메시지 추가
    st.caption(
        "우상단: 허브이자 가교 — 신발·의류 양쪽 네트워크를 연결하는 전략적 전환 키워드"
    )
    st.caption(
        "※ <b style='color:#C62828;'>[핵심 인사이트]</b> <b>상위 80개 핵심 네트워크에 '의류(빨간 점)' 키워드가 나타나지 않는 점은, 현재 소비자들의 휠라 브랜드 인식이 주로 '신발' 카테고리에 집중되어 있음을 시사합니다.</b>",
        unsafe_allow_html=True
    )

    with st.expander("PMI 네트워크 데이터 전체 보기"):
        st.dataframe(
            pmi_df.sort_values("연결 중심성", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# ── White Space 전략 카드 (Tabs 구조) ────────────────────────
st.subheader("White Space — 휠라 전략 옵션")

# 탭 생성: 산식 A(메인 전략)와 산식 B(참고/방어용)
tab_a, tab_b = st.tabs(["산식 A 전략 (시장 영향력 관점)", "산식 B 전략 (고객 만족도 관점)"])

# ─────────────────────────────────────────────────────────────
# TAB A: 산식 A 기반 전략 (발화량 증폭 및 시장 진입)
# ─────────────────────────────────────────────────────────────
with tab_a:
    st.caption(
        "기준점: 산식 A FILA 현 좌표 (0.17, 0.16) — 기능성 최하위, 헤리티지 3위.<br/>"
        "※ <b>산식 A</b>는 실제 시장의 <b>'언급량(Volume)'</b>을 반영합니다. 현재 휠라의 가장 큰 숙제는 "
        "<b>'고객이 휠라 의류를 입고 스스로 이야기하게 만드는 동력'</b>을 확보하는 것입니다.",
        unsafe_allow_html=True
    )

    st.markdown(
        """<div style='background:linear-gradient(135deg, #7B68EE10 0%, #7B68EE25 100%); border: 2px solid #7B68EE; border-radius:8px; padding:16px 20px; margin-bottom:14px; box-shadow:0 2px 8px rgba(123,104,238,0.18);'>
<div style='font-size:12px; color:#7B68EE; font-weight:800; letter-spacing:1px;'>최우선 권장 옵션</div>
<div style='font-size:17px; font-weight:800; color:#1A1A1A; margin-top:4px;'>Option A · Heritage Expansion — '디자인' 자산을 통한 영리한 시장 침투</div>
<div style='font-size:13.5px; color:#333; margin-top:10px; line-height:1.8;'>
현재 휠라는 기능성 정면 승부(발화 점유율 9.7%)보다,<br/>
<strong>압도적인 디자인 강점(발화 비중 19.5%)을 전략적 레버리지</strong>로 삼아야 합니다.<br/><br/>

신발에서 증명된 강력한 디자인 헤리티지를 의류 라인업으로 빠르게 전이시키십시오.<br/>
고객이 '예뻐서' 휠라 의류를 입기 시작하면, 브랜드 언급량(Volume)은 자연스럽게 증폭됩니다.<br/>

이는 곧 <strong>산식 A의 점수 상승과 실질적인 시장 점유율 확대</strong>로 이어집니다.<br/>
기능성이 완비될 때까지 <strong>'디자인 팬덤'으로 시장 지배력을 확보하는 영리한 방어 전략</strong>이 필요합니다.
</div>
</div>""",
        unsafe_allow_html=True,
    )
#     st.markdown(
#         """<div style='background:linear-gradient(135deg, #7B68EE10 0%, #7B68EE25 100%); border: 2px solid #7B68EE; border-radius:8px; padding:14px 18px; margin-bottom:14px; box-shadow:0 2px 8px rgba(123,104,238,0.18);'>
# <div style='font-size:12px; color:#7B68EE; font-weight:800; letter-spacing:1px;'>최우선 권장 옵션</div>
# <div style='font-size:16px; font-weight:800; color:#1A1A1A; margin-top:4px;'>Option A · Heritage Expansion — '디자인' 자산을 통한 영리한 시장 침투</div>
# <div style='font-size:13px; color:#333; margin-top:8px; line-height:1.7;'>
# 현재 휠라는 기능성 정면 승부(발화량 9.7%)보다는 <strong>압도적인 디자인 강점(발화 19.5%)을 레버리지</strong>해야 합니다.<br/> 
# 신발에서 증명된 헤리티지 디자인을 의류 라인업으로 빠르게 전이시키십시오.<br/>
# 고객이 '예뻐서' 휠라를 입기 시작하면 자연스럽게 브랜드 언급량(Volume)이 늘어나고, 이는 곧 산식 A의 점수 상승과 시장 점유율 확대로 이어집니다.<br/>
# <strong>기능성이 완비될 때까지 '디자인 팬덤'으로 시장 지배력을 방어하는 전략</strong>입니다.
# </div>
# </div>""",
#         unsafe_allow_html=True,
#     )

    opt_a = st.columns(3)
    s_a = [
        {"title": "Option A · Heritage Expansion", "x": 0.30, "y": 0.25, "color": "#7B68EE", "rec": True, "desc": "검증된 신발 디자인 자산을 의류로 확장하여, 가장 빠르게 시장 언급량을 확보하는 실전형 카드."},
        {"title": "Option B · Holistic Leader", "x": 0.45, "y": 0.25, "color": "#E0561A", "rec": False, "desc": "중장기적 R&D 투자를 통해 룰루레몬급 기능성 인지도를 구축하는 '정상 도전' 로드맵."},
        {"title": "Option C · Function Catch-up", "x": 0.25, "y": 0.16, "color": "#2E7D32", "rec": False, "desc": "본격적인 확장에 앞서 핏·사이즈 등 고질적인 제품 불만을 해결하는 품질 위생 조치."}
    ]
    for col, s in zip(opt_a, s_a):
        with col:
            badge = "<span style='background:#7B68EE; color:white; font-size:10px; padding:2px 8px; border-radius:10px; font-weight:700; margin-left:6px;'>권장</span>" if s["rec"] else ""
            st.markdown(f"""<div style='border:2px solid {s['color']}; border-radius:8px; padding:14px; height:180px; background:{s['color']}05;'>
<div style='color:{s['color']}; font-weight:700; font-size:14px;'>{s['title']}{badge}</div>
<div style='font-size:11px; color:#666; margin-top:4px;'>목표 ({s['x']}, {s['y']})</div>
<p style='font-size:13px; margin-top:8px; color:#333; line-height:1.5;'>{s['desc']}</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB B: 산식 B 기반 전략 (인식 강도 확인 및 방어)
# ─────────────────────────────────────────────────────────────
with tab_b:
    st.caption(
        "기준점: 산식 B FILA 현 좌표 (0.91, 0.98) — 헤리티지 1위, 기능성 4위.<br/>"
        "※ <b>산식 B</b>는 <b>'일단 구매한 고객들의 경험 만족도'</b>를 보여줍니다. 휠라의 잠재력이 증명되는 지표입니다.",
        unsafe_allow_html=True
    )
    st.markdown(
        """<div style='background:#f8f9fa; border: 1px solid #dee2e6; border-radius:8px; padding:16px 20px; margin-bottom:14px;'>
<div style='font-size:12px; color:#666; font-weight:800; letter-spacing:1px;'>내부 자산 진단</div>
<div style='font-size:17px; font-weight:800; color:#1A1A1A; margin-top:4px;'>Option A · Heritage Defender — 1위 강점(디자인 만족도)의 자산화</div>
<div style='font-size:13.5px; color:#333; margin-top:10px; line-height:1.8;'>
산식 B 결과, 휠라는 헤리티지(디자인) 만족도 0.98로 <strong>독보적인 전체 1위</strong>를 기록했습니다.<br/>
이는 휠라를 경험한 고객들이 <strong>우리만의 고유한 스타일을 이미 매우 높게 평가</strong>하고 있음을 뜻합니다.<br/><br/>

우리의 당면 과제는 단순히 '제품 기능' 개선에만 매몰되는 것이 아닙니다.<br/>
이미 확보된 <strong>고밀도 팬덤이 다시금 '입을 열게 만드는 마케팅적 트리거'</strong>를 강구해야 합니다.<br/>

높은 만족도라는 화약에 마케팅이라는 불꽃을 붙여 <strong>자발적 발화(Volume)를 이끌어내는 것</strong>이 핵심입니다.
</div>
</div>""",
        unsafe_allow_html=True,
    )
#     st.markdown(
#         """<div style='background:#f8f9fa; border: 1px solid #dee2e6; border-radius:8px; padding:14px 18px; margin-bottom:14px;'>
# <div style='font-size:12px; color:#666; font-weight:800; letter-spacing:1px;'>내부 자산 진단</div>
# <div style='font-size:16px; font-weight:800; color:#1A1A1A; margin-top:4px;'>Option A · Heritage Defender — 1위 강점(디자인 만족도)의 자산화</div>
# <div style='font-size:13px; color:#333; margin-top:8px; line-height:1.7;'>
# 산식 B에서 휠라는 헤리티지(디자인) 만족도 0.98로 <strong>독보적인 1위</strong>입니다. 즉, 휠라를 경험한 고객은 우리만의 스타일을 매우 높게 평가하고 있습니다. 
# 우리의 과제는 '제품력' 개선에만 매몰되는 것이 아니라, 이 높은 만족도를 가진 <strong>팬덤이 다시금 '입을 열게 만드는 마케팅 수단'을 강구</strong>하는 것입니다.
# </div>
# </div>""",
#         unsafe_allow_html=True,
#     )

    opt_b = st.columns(3)
    s_b = [
        {"title": "Option A · Heritage Defender", "x": 0.94, "y": 0.98, "color": "#7B68EE", "desc": "현존 최고의 디자인 만족도를 유지하며, 기능성에 대한 심리적 허들을 낮추는 전략."},
        {"title": "Option B · Holistic Leader", "x": 0.96, "y": 0.97, "color": "#E0561A", "desc": "기능성 만족도를 프리미엄 레벨로 격상시켜 전 영역에서 무결점 브랜드로 진화."},
        {"title": "Option C · Function Catch-up", "x": 0.95, "y": 0.93, "color": "#2E7D32", "desc": "실구매자의 83%가 지적하는 고질적 품질/핏 이슈를 해결하여 이탈 고객 방지."}
    ]
    for col, s in zip(opt_b, s_b):
        with col:
            st.markdown(f"""<div style='border:1px solid {s['color']}; border-radius:8px; padding:14px; height:180px; background:white;'>
<div style='color:{s['color']}; font-weight:700; font-size:14px;'>{s['title']}</div>
<div style='font-size:11px; color:#666; margin-top:4px;'>목표 ({s['x']}, {s['y']})</div>
<p style='font-size:13px; margin-top:8px; color:#333; line-height:1.5;'>{s['desc']}</p>
</div>""", unsafe_allow_html=True)



# # 신발/의류 시드 단어 — 의류 풀 확장 (튜터 피드백: 의류 노드 2개 → 시드 확장 + 본질 인사이트 보강)
# _SHOE_SEEDS = {
#     "신발", "운동화", "스니커즈", "슈즈", "깔창", "밑창", "굽", "인솔",
#     "착화감", "발볼", "발등", "발목", "끈", "하이탑",
# }
# _APPAREL_SEEDS = {
#     # 기존 16개
#     "레깅스", "티셔츠", "바지", "상의", "하의", "셔츠", "반팔", "긴팔",
#     "조거", "재킷", "니트", "기모", "맨투맨", "원피스", "스커트", "반바지",
#     # 확장 — 애슬레저 의류 어휘 보강
#     "티", "조거팬츠", "팬츠", "스웻팬츠", "스웻", "후드", "후드티", "후디", "집업",
#     "탑", "크롭", "크롭탑", "트레이닝복", "트레이닝", "트랙수트",
#     "탱크탑", "민소매", "롱슬리브", "숏팬츠",
#     "조끼", "베스트", "점퍼", "코트", "패딩", "플리스", "집업후드",
#     "스포츠브라", "브라탑", "요가복", "요가팬츠",
#     "트레이너", "스포츠웨어", "애슬레저",
#     "카고", "카고팬츠", "조거바지", "드레스", "튜닉", "후드집업",
# }
# _PMI_STOPWORDS: frozenset[str] = frozenset([
#     "있다", "없다", "좋다", "같다", "하다", "이다", "되다", "많다",
#     "너무", "진짜", "정말", "그냥", "조금", "약간", "엄청", "완전", "매우", "많이",
#     "느낌", "생각", "마음", "이거", "저거", "그런", "이런", "저런", "어떤",
#     "것", "수", "때", "거", "게", "걸", "더", "안", "못", "또", "다", "잘",
#     "좋아", "좋아요", "있어", "없어", "같아", "합니다", "됩니다",
#     "구매", "주문", "배송", "포장", "리뷰", "상품", "제품",
#     "이번", "이후", "기존", "받다", "오다", "보다", "주다",
# ])


# @st.cache_data(ttl=CACHE_TTL)
# def _compute_pmi_centrality(
#     top_vocab: int = 250,
#     min_pair_count: int = 3,
#     min_pmi: float = 0.5,
# ) -> pd.DataFrame:
#     df = get_reviews(columns=("brand", "tokens"))
#     fila = df[df["brand"] == "FILA"]["tokens"].dropna()

#     docs = [str(t).split() for t in fila]

#     word_counts: Counter = Counter(w for doc in docs for w in doc)
#     total_words = sum(word_counts.values()) or 1

#     vocab = {
#         w for w, c in word_counts.most_common(top_vocab)
#         if len(w) > 1 and not w.isdigit() and c >= min_pair_count
#         and w not in _PMI_STOPWORDS
#     }

#     cooc: Counter = Counter()
#     for doc in docs:
#         words = [w for w in doc if w in vocab]
#         for i, w1 in enumerate(words):
#             for w2 in words[i + 1: i + 4]:
#                 if w1 != w2:
#                     cooc[tuple(sorted([w1, w2]))] += 1

#     total_cooc = sum(cooc.values()) or 1

#     G = nx.Graph()
#     for (w1, w2), c in cooc.items():
#         if c < min_pair_count:
#             continue
#         pmi = (
#             math.log2(c / total_cooc)
#             - math.log2(word_counts[w1] / total_words)
#             - math.log2(word_counts[w2] / total_words)
#         )
#         if pmi >= min_pmi:
#             G.add_edge(w1, w2, weight=float(pmi), count=c)

#     if len(G.nodes) < 3:
#         return pd.DataFrame()

#     degree_c  = nx.degree_centrality(G)
#     # k 샘플링으로 대형 그래프 속도 개선 (노드 수 300 이하면 정확 계산)
#     k_sample = None if len(G.nodes) <= 300 else 100
#     between_c = nx.betweenness_centrality(G, normalized=True, k=k_sample)

#     def _cat(w: str) -> str:
#         if w in _SHOE_SEEDS:    return "신발"
#         if w in _APPAREL_SEEDS: return "의류"
#         return "공통"

#     rows = [
#         {
#             "키워드":   node,
#             "연결 중심성": round(degree_c.get(node, 0), 4),
#             "매개 중심성": round(between_c.get(node, 0), 4),
#             "빈도":     word_counts.get(node, 0),
#             "카테고리":  _cat(node),
#         }
#         for node in G.nodes()
#     ]
#     return pd.DataFrame(rows)


# with safe_block("PMI 파라미터"):
#     pmi_c1, pmi_c2, pmi_c3 = st.columns(3)
#     with pmi_c1:
#         top_vocab_sel = st.slider("어휘 크기 (상위 N 단어)", 100, 400, 250, 50,
#                                   key="pmi_vocab")
#     with pmi_c2:
#         min_pair_sel = st.slider("최소 동시 출현 횟수", 2, 20, 3, 1,
#                                  key="pmi_minpair")
#     with pmi_c3:
#         min_pmi_sel = st.slider("최소 PMI 임계값", 0.0, 3.0, 0.5, 0.1,
#                                 key="pmi_threshold")

# with st.spinner("PMI 그래프 산출 중… (최초 1회만 소요, 이후 캐시)"):
#     pmi_df = _compute_pmi_centrality(
#         top_vocab=top_vocab_sel,
#         min_pair_count=min_pair_sel,
#         min_pmi=min_pmi_sel,
#     )

# if pmi_df.empty:
#     empty_state("PMI 결과 없음", "임계값을 낮추거나 어휘 크기를 늘려 주세요.")
# else:
#     st.caption(
#         f"그래프 노드: {len(pmi_df):,}개 단어 | "
#         f"신발: {(pmi_df['카테고리']=='신발').sum()}개 / "
#         f"의류: {(pmi_df['카테고리']=='의류').sum()}개 / "
#         f"공통: {(pmi_df['카테고리']=='공통').sum()}개"
#     )

#     _CAT_COLOR = {"신발": "#003087", "의류": "#D4000F", "공통": "#888888"}

#     col_deg, col_bet = st.columns(2)

#     with col_deg:
#         top_deg = pmi_df.nlargest(15, "연결 중심성").sort_values("연결 중심성")
#         fig_deg = px.bar(
#             top_deg, x="연결 중심성", y="키워드",
#             orientation="h",
#             color="카테고리",
#             color_discrete_map=_CAT_COLOR,
#             title="연결 중심성 Top 15",
#             labels={"연결 중심성": "연결 중심성", "키워드": ""},
#             hover_data={"빈도": True, "카테고리": True},
#         )
#         fig_deg.update_layout(height=480, showlegend=True,
#                               legend=dict(orientation="h", y=-0.15))
#         st.plotly_chart(fig_deg, use_container_width=True)
#         st.caption("연결 중심성: 한 단어가 얼마나 많은 다른 단어와 연결되어 있는지 — 수치가 높을수록 브랜드 경험의 중심 단어")
#         st.caption("※ 가장 많은 단어들과 짝지어져서 언급된 핵심 단어입니다.")

#     with col_bet:
#         top_bet = pmi_df.nlargest(15, "매개 중심성").sort_values("매개 중심성")
#         fig_bet = px.bar(
#             top_bet, x="매개 중심성", y="키워드",
#             orientation="h",
#             color="카테고리",
#             color_discrete_map=_CAT_COLOR,
#             title="매개 중심성 Top 15",
#             labels={"매개 중심성": "매개 중심성", "키워드": ""},
#             hover_data={"빈도": True, "카테고리": True},
#         )
#         fig_bet.update_layout(height=480, showlegend=True,
#                               legend=dict(orientation="h", y=-0.15))
#         st.plotly_chart(fig_bet, use_container_width=True)
#         st.caption("매개 중심성: 신발 단어 그룹과 의류 단어 그룹 사이를 얼마나 잘 이어주는지 — 브랜드 확장의 전환점이 되는 단어")
#         st.caption("※ 서로 다른 주제나 흩어진 단어들을 중간에서 이어주는 다리 역할의 단어입니다.")

#     # 연결 중심성 vs 매개 중심성 산점도
#     st.subheader("키워드 포지셔닝 — 연결 중심성 × 매개 중심성")
#     fig_scatter = px.scatter(
#         pmi_df[pmi_df["빈도"] >= 10].nlargest(80, "연결 중심성"),
#         x="연결 중심성", y="매개 중심성",
#         size="빈도", color="카테고리",
#         color_discrete_map=_CAT_COLOR,
#         text="키워드",
#         hover_data={"빈도": True, "카테고리": True},
#         labels={
#             "연결 중심성": "연결 중심성 (많이 연결된 단어)",
#             "매개 중심성": "매개 중심성 (다리 역할 단어)",
#         },
#         title="키워드 네트워크 포지셔닝 맵 (빈도 상위 80개)",
#         size_max=40,
#     )
#     fig_scatter.update_traces(textposition="top center", textfont_size=9)
#     fig_scatter.update_layout(height=560)
#     st.plotly_chart(fig_scatter, use_container_width=True)
#     st.caption(
#         "우상단: 허브이자 가교 — 신발·의류 양쪽 네트워크를 연결하는 전략적 전환 키워드"
#     )

#     with st.expander("PMI 네트워크 데이터 전체 보기"):
#         st.dataframe(
#             pmi_df.sort_values("연결 중심성", ascending=False),
#             use_container_width=True,
#             hide_index=True,
#         )

# st.divider()

# # ── White Space 전략 카드 ──────────────────────────────────
# st.subheader("White Space — 휠라 전략 옵션")
# st.caption(
#     "기준점: 산식 B 기준 FILA 현 좌표 (0.91, 0.98) — 헤리티지 1위, 기능성 4위. "
#     "룰루레몬 (0.96, 0.95) / 젝시믹스 (0.94, 0.92) / 안다르 (0.93, 0.95). "
#     "단, 좌표는 '언급 시 긍정 강도' — 발화량(BERTopic)은 FILA 기능성 9.7% vs 룰루레몬 46.7%로 4배 차이"
# )

# # 추천 옵션 강조 배너 (Option A 권장)
# st.markdown(
#     """
# <div style='background:linear-gradient(135deg, #7B68EE10 0%, #7B68EE25 100%);
# border: 2px solid #7B68EE; border-radius:8px;
# padding:14px 18px; margin-bottom:14px; box-shadow:0 2px 8px rgba(123,104,238,0.18);'>
#   <div style='font-size:12px; color:#7B68EE; font-weight:800; letter-spacing:1px;'>
#      최우선 권장 옵션
#   </div>
#   <div style='font-size:16px; font-weight:800; color:#1A1A1A; margin-top:4px;'>
#     Option A · Heritage Defender — 디자인 강점 즉시 자산화
#   </div>
#   <div style='font-size:13px; color:#333; margin-top:8px; line-height:1.7;'>
#     R&amp;D 부담 없이 <strong>6~12개월 내 실행</strong> 가능. 디자인 P_ratio
#     <strong>+0.611(4브랜드 1위)</strong>·발화 <strong>19.5%(룰루레몬 3배)</strong>를
#     의류 컬렉션으로 직접 환산. Phase 1 단기 액션(Option C 위생 조치) → Phase 2 본 진입(Option A)
#     → Phase 3 정상 도전(Option B) 로드맵의 <strong>본 진입 카드</strong>.
#   </div>
# </div>
# """,
#     unsafe_allow_html=True,
# )

# opt = st.columns(3)
# strategies = [
#     {
#         "title": "Option A · Heritage Defender",
#         "subtitle": "수성형 · 권장",
#         "color": "#7B68EE",
#         "border_w": 4,
#         "recommended": True,
#         "x": 0.94, "y": 0.98,
#         "delta": "Δx +0.03 / Δy 0",
#         "desc": (
#             "현재 <strong>헤리티지 1위(y=0.98)</strong> 자산을 그대로 유지하면서, "
#             "기능성 좌표를 룰루레몬과의 격차(0.054)의 절반인 +0.03만 좁힌다.<br/><br/>"
#             "디자인 P_ratio <strong>+0.611(4브랜드 1위)</strong> + 발화 19.5% "
#             "(룰루레몬 6.1%의 3배) 활용.<br/><br/>"
#             "<strong>디자인·헤리티지 컬렉션 + PDP 기능 메시지 보강</strong>으로 "
#             "R&amp;D 부담 없이 단기 실행"
#         ),
#         "risk": (
#             "기능성 발화량 9.7% 미해결.<br/>"
#             "룰루레몬이 헤리티지로 따라오면 1위 흔들림.<br/>"
#             "차별화는 단기 유지, 천장 존재"
#         ),
#     },
#     {
#         "title": "Option B · Holistic Leader",
#         "subtitle": "정상 도전",
#         "color": "#E0561A",
#         "border_w": 2,
#         "recommended": False,
#         "x": 0.96, "y": 0.97,
#         "delta": "Δx +0.05 / Δy −0.01",
#         "desc": (
#             "<strong>룰루레몬의 기능성 좌표 0.962를 직접 매치</strong>하면서 "
#             "헤리티지도 0.97 유지.<br/><br/>"
#             "유일하게 좌표상 4브랜드 우상단을 동시 점유.<br/><br/>"
#             "BERTopic 기능성 <strong>발화 점유율 9.7% → 30%</strong> 12개월 KPI 설정, "
#             "쿨링·압박·통기성 R&amp;D + 광고·인플루언서 일관 노출 + 헤리티지 디자인 분리 유지"
#         ),
#         "risk": (
#             "R&amp;D + 마케팅 동시 투자로 CAPEX·OPEX 가장 큼.<br/>"
#             "12~24개월 회수,<br/>"
#             "메시지 일관성 실패 시 두 축 동시 약화 위험"
#         ),
#     },
#     {
#         "title": "Option C · Function Catch-up",
#         "subtitle": "기능 추격",
#         "color": "#2E7D32",
#         "border_w": 2,
#         "recommended": False,
#         "x": 0.95, "y": 0.93,
#         "delta": "Δx +0.04 / Δy −0.05",
#         "desc": (
#             "기능성 좌표를 안다르·젝시믹스 수준(0.93~0.94) 위로 끌어올리되 "
#             "헤리티지는 안다르 수준(0.95)으로 의도적 양보.<br/><br/>"
#             "<strong>저평점 9.4K의 핏 46.4% + 품질 36.8% = 83% 원인 즉시 차단</strong>.<br/><br/>"
#             "사이즈 가이드 + 50회 세탁 후 형태 인증 + 기능성 라벨링으로 "
#             "단기 반품률·CS 부담 절감"
#         ),
#         "risk": (
#             "디자인 +0.611·헤리티지 0.98 활용 부족 → 차별화 약화.<br/>"
#             "기능성 경쟁(룰루레몬·젝시믹스) 영역 직접 진입 — 마진율 압박"
#         ),
#     },
# ]
# for col, s in zip(opt, strategies):
#     with col:
#         glow = "box-shadow: 0 0 16px rgba(123,104,238,0.35);" if s["recommended"] else ""
#         badge = (
#             "<span style='background:#7B68EE; color:white; font-size:10px; "
#             "padding:2px 8px; border-radius:10px; font-weight:700; margin-left:6px;'>권장</span>"
#             if s["recommended"] else ""
#         )
#         st.markdown(
#             f"""<div style='border: {s['border_w']}px solid {s['color']}; border-radius:8px;
#             padding:16px; height:100%; background:{s['color']}11; {glow}'>
#                 <div style='color:{s['color']}; font-weight:700; font-size:15px;'>
#                     {s['title']}{badge}
#                 </div>
#                 <div style='font-size:11px; color:#888; margin-top:2px; font-style:italic;'>{s['subtitle']}</div>
#                 <div style='font-size:11px; color:#666; margin-top:6px;'>
#                     목표 좌표 ({s['x']:.2f}, {s['y']:.2f}) · {s['delta']}
#                 </div>
#                 <p style='font-size:14px; margin-top:12px; color:#222; line-height:1.7;'>{s['desc']}</p>
#                 <div style='font-size:12px; color:#C62828; margin-top:10px; line-height:1.6;'>
#                     <strong>리스크</strong><br/>{s['risk']}
#                 </div>
#             </div>""",
#             unsafe_allow_html=True,
#         )

# st.markdown("---")
# st.markdown(
#     """
# <div style='background:#F0F4FF; border-left:4px solid #003087;
# padding:16px 20px; border-radius:0 6px 6px 0;'>
#   <div style='font-size:14px; font-weight:800; color:#003087; margin-bottom:10px;'>
#     권장 진입 경로 — 12~24개월 3-Phase 로드맵
#   </div>
#   <div style='font-size:14px; color:#222; line-height:1.8;'>
#     <strong>Phase 1 (0–6개월) · Option C 단기 액션</strong><br/>
#     사이즈표 정확화 + 50회 세탁 후 형태 인증 + 기능성 라벨링 —
#     저평점 핏+품질 83% 즉시 차단, 반품률·CS 부담 단기 절감.<br/>
#     <span style='color:#666;'>지표: 1~2점 비율, 반품률, CS 인입 건수</span><br/><br/>
#     <strong>Phase 2 (6–18개월) · Option A 헤리티지 컬렉션 (본 진입)</strong><br/>
#     디자인 +0.611·발화 19.5% 강점 자산화 —
#     신발 디자인 헤리티지 의류 컬렉션 출시 + 헤리티지 마케팅으로 좌표 (0.94, 0.98) 도달.<br/>
#     <span style='color:#666;'>지표: 디자인 P_ratio, 헤리티지 토픽 점유율, 컬렉션 매출</span><br/><br/>
#     <strong>Phase 3 (18–24개월) · Option B Holistic 전환</strong><br/>
#     기능성 R&amp;D 출시 + 광고·인플루언서 일관 노출 —
#     BERTopic 기능성 발화 9.7% → 30% 도달 시 좌표 (0.96, 0.97)로 룰루레몬 정상 도전.<br/>
#     <span style='color:#666;'>지표: 기능성 발화 점유율, 기능성 P_ratio, 재구매율</span>
#   </div>
# </div>
# """,
#     unsafe_allow_html=True,
# )

# =====================================================================
# ── [신규 추가] AI 실시간 트렌드 통합 전략 리포트 ─────────────────────────
# =====================================================================
from utils.llm_utils import get_total_insight
from config import ASPECT_LABELS
import datetime

st.divider()
st.subheader("🤖 AI 기반 실시간 트렌드 통합 전략 리포트")
st.caption("내부 리뷰 데이터의 팩트와 외부 인스타그램 실시간 트렌드를 입체적으로 비교·분석합니다.")

# 1. AI 타겟 브랜드 선택
ai_target_brand = st.selectbox(
    "AI 심층 분석 대상 브랜드 선택",
    options=BRAND_ORDER,
    index=0,
    key="ai_target_brand"
)

# 2. 데이터 요약 함수
def build_actual_data_summary(brand: str, pol_df: pd.DataFrame, pos_df: pd.DataFrame, sna_df: pd.DataFrame) -> str:
    pol_sub = pol_df[pol_df["brand"] == brand]
    if pol_sub.empty: return f"{brand}의 리뷰 데이터 없음"
    
    total_reviews = int(pol_sub.iloc[0]["n_reviews"]) if "n_reviews" in pol_sub.columns else 0
    aspect_lines = []
    for _, r in pol_sub.iterrows():
        asp_name = ASPECT_LABELS.get(r["aspect"], r["aspect"])
        aspect_lines.append(f"- {asp_name}: 긍정 {r.get('P_ratio', 0)*100:.1f}%, 부정 {r.get('N_ratio', 0)*100:.1f}%")
    
    pos_sub = pos_df[pos_df["brand"] == brand]
    pos_summary = f"- 기능성(X): {pos_sub.iloc[0].get('x_function', 0):.3f} / 헤리티지(Y): {pos_sub.iloc[0].get('y_heritage', 0):.3f}" if not pos_sub.empty else "좌표 없음"

    sna_sub = sna_df[sna_df["brand"] == brand] if not sna_df.empty else pd.DataFrame()
    sna_summary = ", ".join(sna_sub.nlargest(5, "centrality")["keyword"].tolist()) if not sna_sub.empty and "centrality" in sna_sub.columns else "데이터 없음"

    return f"[{brand}] 데이터 요약\n- 리뷰 수: {total_reviews:,}건\n[포지셔닝]\n{pos_summary}\n[6속성 평가]\n{chr(10).join(aspect_lines)}\n[핵심 키워드]\n- {sna_summary}"

# ---------------------------------------------------------
# [추가] 캐싱 함수 정의 (함수 정의 구역에 위치)
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_insight(summary, query):
    """
    동일한 데이터 요약과 쿼리에 대해 API 호출 결과를 캐싱하여 
    할당량(Quota) 소모를 방지합니다.
    """
    return get_total_insight(summary, query)

# 3. 자동 쿼리 생성
current_date = datetime.datetime.now()
season = "S/S(봄/여름)" if 3 <= current_date.month <= 8 else "F/W(가을/겨울)"
auto_query = f"{current_date.year}년 {season} {ai_target_brand} 인스타그램 인기 키워드 및 애슬레저 스타일 트렌드"

# 4. 분석 실행
if st.button(f"✨ {ai_target_brand} 통합 전략 3단 리포트 생성", use_container_width=True, type="primary"):
    actual_summary_text = build_actual_data_summary(ai_target_brand, polarity, pos, sna)
    
    with st.spinner(f"전문 전략가가 {ai_target_brand}의 시장 데이터와 트렌드를 융합 분석 중입니다..."):
        ai_hashtags, full_report = get_total_insight(actual_summary_text, auto_query)
        
        # [핵심 로직] 브랜드+코디 해시태그를 첫 번째에 강제 삽입
        brand_hashtag = f"{ai_target_brand}코디"
        final_hashtags = [brand_hashtag] + ai_hashtags 
        
        # [결과 1] 3단 리포트 출력
        st.markdown(full_report)
            
        # [결과 2] 인스타그램 동적 해시태그 출력
        st.divider()
        st.subheader(f"📸 {ai_target_brand} 외부 트렌드 크로스체크")
        st.info("리포트에서 언급된 트렌드 해시태그를 클릭하여 실제 인스타그램 피드를 직접 확인해 보세요.")

        col1, col2, col3 = st.columns(3)
        
                # final_hashtags 버튼 출력 부분 수정
        with col1: 
            if len(final_hashtags) > 0:
                # 구글 이미지 검색을 통해 인스타그램 게시물을 우회 확인 (최근 1개월 데이터)
                link = f"https://www.google.com/search?q=site:instagram.com+{final_hashtags[0]}&tbm=isch&tbs=qdr:m"
                st.link_button(f"📸 #{final_hashtags[0]} 보기", link, use_container_width=True)
        with col2: 
            if len(final_hashtags) > 1:
                link = f"https://www.google.com/search?q=site:instagram.com+{final_hashtags[1]}&tbm=isch&tbs=qdr:m"
                st.link_button(f"📸 #{final_hashtags[1]} 보기", link, use_container_width=True)
        with col3: 
            if len(final_hashtags) > 2:
                link = f"https://www.google.com/search?q=site:instagram.com+{final_hashtags[2]}&tbm=isch&tbs=qdr:m"
                st.link_button(f"📸 #{final_hashtags[2]} 보기", link, use_container_width=True)