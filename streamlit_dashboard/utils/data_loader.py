"""
data_loader.py — 통합 데이터 로더 + 더미 생성기
================================================

설계 원칙 (Decoupled Architecture):
- 페이지·컴포넌트는 절대 parquet 경로를 직접 알지 못한다.
- get_reviews(), get_topics(), get_absa(), get_positioning(), get_sna() 만 호출.
- 실제 파일이 존재하면 → 로드 + 스키마 검증.
- 존재하지 않으면 → 결정론적(seed) 더미를 즉시 생성하여 동일 인터페이스로 반환.
- 모델팀이 Parquet을 떨어뜨리는 순간, 페이지 코드 변경 없이 실데이터로 전환.

캐시 전략:
- 모든 로더는 @st.cache_data(ttl=CACHE_TTL).
- 인자 단위로 캐시 키 생성 → 필터 조합별 결과 보존.
- 117만 건 reviews 는 columns 인자로 사용 컬럼만 로드 (Parquet 컬럼 프루닝).
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import math
import networkx as nx
from itertools import combinations
import numpy as np
import pandas as pd
import streamlit as st
import os
import gdown

from config import (
    PATHS, CACHE_TTL, BRAND_ORDER, ASPECT_KEYS, SENTIMENT_LABELS,
    MIN_REVIEWS_FOR_BRAND_SCORE, BERT_ASPECT_KR,
)
from utils.data_contracts import (
    REVIEWS_SCHEMA, TOPICS_SCHEMA, TOPIC_META_SCHEMA,
    ABSA_SCHEMA, POSITIONING_SCHEMA, SNA_SCHEMA,
    check_schema,
)
from utils.exceptions import warn_using_dummy

logger = logging.getLogger(__name__)

DUMMY_SEED = 42
DUMMY_REVIEW_N = 5_000   # 더미 모드에서 생성할 리뷰 수

# SNA 공출현 그래프에서 제외할 일반 불용어
_SNA_STOPWORDS: frozenset[str] = frozenset([
    # 사용자 지정
    "좋다", "같다", "구매", "많다", "생각", "제품", "맞다", "마음",
    "하다", "있다", "없다", "너무", "진짜", "이거", "저거", "그냥",
    "정도", "조금", "약간", "느낌", "리뷰", "상품",
    # 기본 동사·형용사
    "이다", "되다", "받다", "오다", "보다", "주다", "알다", "가다",
    "좋아", "좋아요", "괜찮다", "괜찮아", "예쁘다", "이쁘다",
    "있어", "없어", "같아", "되게", "되어", "됩니다", "합니다",
    # 부사·감탄사
    "정말", "엄청", "완전", "매우", "많이", "역시", "딱", "아주",
    "좀", "더", "안", "못", "또", "다", "잘", "꽤", "참",
    # 지시대명사·접속사
    "그런", "이런", "저런", "어떤", "그리고", "하지만", "그러나",
    "근데", "그래도", "그래서", "그냥요", "근데요",
    # 의존명사·불완전 형태소
    "것", "수", "때", "거", "게", "걸", "뭔가", "거나",
    "이번", "이후", "기존", "사서", "주문", "배송", "포장",
])


# ═════════════════════════════════════════════════════════════
# 1. Reviews — 실데이터 우선, 없으면 더미
# ═════════════════════════════════════════════════════════════
@st.cache_data(ttl=CACHE_TTL, show_spinner="리뷰 데이터 로드 중...")
def get_reviews(columns: tuple[str, ...] | None = None,
                sample_n: int | None = None) -> pd.DataFrame:
    """리뷰 마스터 로드.

    Args:
        columns: 사용할 컬럼만 로드 (Parquet 컬럼 프루닝). None=전체.
        sample_n: 빠른 EDA용 다운샘플. None=전체.
    """
    path = PATHS["reviews"]
    if path.exists():
        cols = list(columns) if columns else None
        df = pd.read_parquet(path, columns=cols)
        # 부분 컬럼 로드 시 스키마 체크 생략 (누락 컬럼 오경보 방지)
        if cols is None:
            result = check_schema(df, REVIEWS_SCHEMA, "reviews")
            if not result.ok:
                logger.warning(result.summary())
        if sample_n and len(df) > sample_n:
            df = df.sample(sample_n, random_state=DUMMY_SEED)
        return df

    # 더미 폴백
    logger.info(f"[DUMMY] {path.name} 미발견 → 더미 생성")
    return _generate_dummy_reviews(DUMMY_REVIEW_N if not sample_n else sample_n)


@st.cache_data(ttl=CACHE_TTL)
def get_topics() -> pd.DataFrame:
    return _generate_dummy_topics()


@st.cache_data(ttl=CACHE_TTL, show_spinner="토큰 데이터 로드 중...")
def get_tokens(brand: str | None = None,
               column: str = "tokens_topic") -> pd.DataFrame:
    """reviews 마스터에서 형태소 토큰 컬럼 로드."""
    path = PATHS["reviews"]
    if not path.exists():
        return pd.DataFrame(columns=["brand", column])
    try:
        df = pd.read_parquet(path, columns=["brand", column])
    except Exception:
        # 컬럼 부재 시 빈 리스트 컬럼으로 대체
        df = pd.read_parquet(path, columns=["brand"])
        df[column] = [[] for _ in range(len(df))]
    if brand is not None:
        df = df[df["brand"] == brand]
    return df


@st.cache_data(ttl=CACHE_TTL)
def get_topic_meta() -> pd.DataFrame:
    """토픽 메타(topic_id별 1행). topics에서 파생."""
    topics = get_topics()

    if topics.empty:
        return pd.DataFrame(columns=list(TOPIC_META_SCHEMA.keys()))

    # BERTopic 실 산출물은 'Topic'(대문자) 또는 'topic'(소문자) 컬럼을 사용할 수 있음
    for _old in ("Topic", "topic"):
        if "topic_id" not in topics.columns and _old in topics.columns:
            topics = topics.rename(columns={_old: "topic_id"})
            break

    # keyword_1~keyword_5 컬럼이 있으면 리스트로 합쳐 topic_keywords 생성
    _kw_cols = [f"keyword_{i}" for i in range(1, 6) if f"keyword_{i}" in topics.columns]
    if _kw_cols and "topic_keywords" not in topics.columns:
        topics["topic_keywords"] = topics[_kw_cols].apply(
            lambda r: [str(v).strip() for v in r if pd.notna(v) and str(v).strip()], axis=1
        )

    if "topic_id" not in topics.columns:
        warn_using_dummy("토픽 메타데이터 (컬럼 누락)")
        rng = np.random.default_rng(DUMMY_SEED + 10)
        dummy_topics = [
            ("쿠셔닝/착용감", ["쿠션", "폭신", "착용감", "안정감"]),
            ("핏/보정",       ["핏", "사이즈", "라인", "보정"]),
            ("디자인/색감",   ["디자인", "색상", "컬러", "패턴"]),
            ("소재/촉감",     ["소재", "촉감", "원단", "신축성"]),
            ("재구매/추천",   ["재구매", "추천", "단골", "믿고"]),
        ]
        return pd.DataFrame([
            {
                "topic_id":   i,
                "topic_name": name,
                "n_reviews":  int(rng.integers(500, 5000)),
                "keywords":   kws,
                "axis_hint":  _axis_hint_for_topic(name),
                "representative_doc": "",
            }
            for i, (name, kws) in enumerate(dummy_topics)
        ])

    agg_kwargs: dict = {"topic_name": ("topic_name", "first")} if "topic_name" in topics.columns else {}
    id_col = "review_id" if "review_id" in topics.columns else topics.columns[0]
    agg_kwargs["n_reviews"] = (id_col, "count")
    if "topic_keywords" in topics.columns:
        agg_kwargs["keywords"] = ("topic_keywords", "first")
    elif "topic_name" in topics.columns:
        # keyword 컬럼이 없으면 topic_name을 단어로 분해해 keywords 대체
        topics["_kw_fallback"] = topics["topic_name"].apply(lambda n: n.split("/") if n else [])
        agg_kwargs["keywords"] = ("_kw_fallback", "first")

    meta = topics.groupby("topic_id", as_index=False).agg(**agg_kwargs)

    if "topic_name" not in meta.columns:
        meta["topic_name"] = meta["topic_id"].astype(str)
    if "keywords" not in meta.columns:
        meta["keywords"] = [[] for _ in range(len(meta))]

    meta["axis_hint"] = meta["topic_name"].apply(_axis_hint_for_topic)
    meta["representative_doc"] = ""
    return meta


_ABSA_COL_RENAME = {
    "핏/사이즈":      "fit_size",
    "소재/내구성":    "material_durability",
    "기능성":         "functionality",
    "디자인":         "design",
    "브랜드/헤리티지": "brand_heritage",
    "가격/가치":      "price_value",
}

def _load_absa_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    rename_map = {k: v for k, v in _ABSA_COL_RENAME.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@st.cache_data(ttl=CACHE_TTL)
def get_absa(sample_mode: str = "all") -> pd.DataFrame:
    """ABSA phase_e 최종 결과 로드 (4브랜드 × 3,014건)."""
    path = PATHS["absa"]
    if path.exists():
        df = _load_absa_parquet(path)
        # confidence 컬럼 누락 시 NaN으로 동적 보완 (스키마 경고 없이 로드 보장)
        for aspect in ASPECT_KEYS:
            conf_col = f"{aspect}_confidence"
            if conf_col not in df.columns:
                df[conf_col] = np.nan
        logger.info(f"[ABSA] phase_e 로드: {len(df):,}")
        return df
    logger.info("[DUMMY] absa 파일 미발견 → 더미 생성")
    return _generate_dummy_absa()


@st.cache_data(ttl=CACHE_TTL)
def get_positioning() -> pd.DataFrame:
    """ABSA → 브랜드 포지셔닝 좌표 동적 산출."""
    return compute_positioning_from_absa()


# ─────────────────────────────────────────────────────────────
# [사전 정의] PMI 네트워크 고도화를 위한 사용자 사전 및 불용어
# ─────────────────────────────────────────────────────────────
ADVANCED_STOPWORDS = {
    '네이버', '페이', '후기', '작성', '등록', '포인트', '아울렛', '로그인', '사이트', '구매', '제품', '상품', '주문', '배송',
    '이번', '선택', '사용', '생각', '평소', '하다', '되다', '들르다', '들렀다', '특유', '햇빛', '조명', '따르다', '시선',
    '만들다', '측면', '덕분', '신다', '더하다', '필수', '쟁이다', '그냥', '아웃렛', '와이프', '않다', '안다', '내다', '같다',
    '울다', '그렇다', '이렇다', '어떻다', '저렇다', '사다', '입다', '순간', '구입', '종류', '달라다', '넘다', '부분', '알다',
    '보이다', '중요', '보다', '기준', '올리다', '내리다', '찾다', '장난', '죽다', '생명', '덥다', '힘들다', '삐지다', '받다',
    '적다', '착용', '닿다', '타사', '나오다', '넣다', '방식', '세일', '쎄일', '안파다', '배송비', '살다', '할인', '기분', '디다',
    '감안', '안나오다', '안입다', '요즘', '요즈음', '역쉬', '역시', '기다', '몰다', '모르다', '돌리다', '바람', '특가', '행사',
    '가리다', '리뷰', '안타다', '타다', '나가다', '쓰다', '딸아이', '아이', '아들', '남편', '아내', '부인', '재다', '동생',
    '언니', '누나', '오빠', '형', '문의', '교환', '감수', '입지', '나다', '치다', '박다', '짙다', '베란다', '일주일', '두다',
    '하이', '일반', '비하다', '비슷', '사람', '물어보다', '시즌', '날씨', '껴입다', '딱오다', '도움', '기부니', '차이', '돌다',
    '품목', '시키다', '전체', '느끼다', '휠라', '젝시믹스', '안다르', '룰루레몬', '진짜', '너무', '정도', '많이', '조금', '약간'
}

# 💡 핵심: 의류 확장을 증명하기 위해 빈도수와 무관하게 차트에 무조건 살려둘 시드 단어들
MUST_INCLUDE_SEEDS = {
    # 1. 휠라 헤리티지 & 최신 트렌드 견인 (신발/핵심 라인업)
    '인터런', '디스럽터', '에샤페', '레이', '페이토', '오크먼트', '코트디럭스', 
    '메리제인', '글리오', '하레핀', '레인저코어', '테니스화',
    
    # 2. 일반 스포츠/캐주얼 의류 (기존 휠라가 가진 의류 인식)
    '조거팬츠', '스웻', '아노락', '맨투맨', '트랙탑', '트랙팬츠', '바람막이', '티셔츠', '팬츠', '쇼츠',
    
    # 3. 🎯 핵심 타겟: 애슬레저 코어 아이템 (안다르/룰루레몬 파이 뺏기 용도)
    '레깅스', '브라탑', '스포츠브라', '부츠컷', '부츠컷레깅스', '바이커쇼츠', '숏팬츠', '크롭티', '커버업'
}

@st.cache_data(ttl=CACHE_TTL, show_spinner="PMI 네트워크(SNA) 연산 중...")
def get_sna(top_n_words: int = 150) -> pd.DataFrame:
    token_col = "tokens_topic"
    df = get_reviews(columns=("brand", token_col))
    if df.empty or token_col not in df.columns:
        token_col = "tokens"
        df = get_reviews(columns=("brand", token_col))
    if df.empty or token_col not in df.columns:
        return _generate_dummy_sna()

    rows = []
    for brand, sub in df.groupby("brand", observed=True):
        total_docs = len(sub)
        doc_freq: dict[str, int] = {}
        
        for toks in sub[token_col]:
            if not isinstance(toks, (list, np.ndarray)):
                continue
            unique_toks = set(w for w in toks if isinstance(w, str) and len(w) >= 2 and w not in ADVANCED_STOPWORDS)
            for w in unique_toks:
                doc_freq[w] = doc_freq.get(w, 0) + 1

        if not doc_freq:
            continue

        top_150 = sorted(doc_freq, key=lambda x: -doc_freq[x])[:top_n_words]
        top_words = set(top_150) | {w for w in MUST_INCLUDE_SEEDS if w in doc_freq}

        co_freq = {}
        for toks in sub[token_col]:
            if not isinstance(toks, (list, np.ndarray)):
                continue
            words_in_review = list({w for w in toks if w in top_words})
            for w1, w2 in combinations(words_in_review, 2):
                pair = tuple(sorted([w1, w2]))
                co_freq[pair] = co_freq.get(pair, 0) + 1

        G: nx.Graph = nx.Graph()
        for node in top_words:
            G.add_node(node)

        for (w1, w2), co_count in co_freq.items():
            if co_count >= 2:
                pmi = math.log2((co_count * total_docs) / (doc_freq[w1] * doc_freq[w2]))
                if pmi > 0:
                    G.add_edge(w1, w2, weight=pmi, distance=1.0/pmi, co_occur=co_count)

        if G.number_of_nodes() == 0:
            continue

        degree = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight="distance", normalized=True)

        for word in G.nodes():
            rows.append({
                "keyword":                word,
                "brand":                  brand,
                "centrality":             round(degree.get(word, 0.0), 4),
                "betweenness_centrality": round(betweenness.get(word, 0.0), 4),
                "frequency":              doc_freq.get(word, 0),
                "polarity":               0.0,
                "topic_id":               -1,
            })

    if not rows:
        return _generate_dummy_sna()

    return pd.DataFrame(rows)

def _generate_dummy_sna() -> pd.DataFrame:
    """더미 데이터 생성 시 KeyError 방지를 위해 betweenness_centrality 컬럼 추가"""
    rng = np.random.default_rng(DUMMY_SEED + 3)
    keywords_pool = [
        "쿠션", "착용감", "쫀쫀", "통기성", "보정", "허리", "재구매", "추천", "디자인",
        "색감", "가성비", "사이즈", "기능성", "신축성", "발편안", "운동",
    ]
    rows = []
    for b in BRAND_ORDER:
        for kw in keywords_pool:
            rows.append({
                "keyword":    kw,
                "brand":      b,
                "centrality": float(rng.uniform(0.05, 0.95)),
                "betweenness_centrality": float(rng.uniform(0.01, 0.5)), # 💡 누락되었던 컬럼 추가 완료
                "topic_id":   int(rng.integers(0, 8)),
                "frequency":  int(rng.integers(50, 5000)),
                "polarity":   float(rng.uniform(-0.4, 0.9)),
            })
    return pd.DataFrame(rows)

# @st.cache_data(ttl=CACHE_TTL, show_spinner="SNA 연산 중...")
# def get_sna(top_n_words: int = 150) -> pd.DataFrame:
#     """reviews 형태소 토큰 → NetworkX 공출현 그래프 → 연결/매개 중심성 산출.

#     top_n_words: 브랜드별 고빈도 상위 N개 단어만 그래프에 포함 (연산 부하 제한).
#     반환 컬럼: keyword, brand, centrality(연결), betweenness_centrality(매개), frequency, polarity, topic_id
#     """
#     import networkx as nx
#     from itertools import combinations

#     token_col = "tokens_topic"
#     df = get_reviews(columns=("brand", token_col))
#     if df.empty or token_col not in df.columns:
#         token_col = "tokens"
#         df = get_reviews(columns=("brand", token_col))
#     if df.empty or token_col not in df.columns:
#         return _generate_dummy_sna()

#     rows = []
#     for brand, sub in df.groupby("brand", observed=True):
#         # Pass 1: 단어 빈도 집계 (불용어 제외)
#         word_freq: dict[str, int] = {}
#         for toks in sub[token_col]:
#             if not isinstance(toks, (list, np.ndarray)):
#                 continue
#             for w in toks:
#                 if (isinstance(w, str) and len(w) >= 2
#                         and w not in _SNA_STOPWORDS):
#                     word_freq[w] = word_freq.get(w, 0) + 1

#         if not word_freq:
#             continue

#         top_words = set(sorted(word_freq, key=lambda x: -word_freq[x])[:top_n_words])

#         # Pass 2: 공출현 그래프 구축
#         G: nx.Graph = nx.Graph()
#         for node in top_words:
#             G.add_node(node)

#         for toks in sub[token_col]:
#             if not isinstance(toks, (list, np.ndarray)):
#                 continue
#             words_in_review = list({w for w in toks if isinstance(w, str) and w in top_words})
#             for w1, w2 in combinations(words_in_review, 2):
#                 if G.has_edge(w1, w2):
#                     G[w1][w2]["weight"] += 1
#                 else:
#                     G.add_edge(w1, w2, weight=1)

#         if G.number_of_nodes() == 0:
#             continue

#         degree = nx.degree_centrality(G)
#         betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

#         for word in G.nodes():
#             rows.append({
#                 "keyword":                word,
#                 "brand":                  brand,
#                 "centrality":             round(degree.get(word, 0.0), 4),
#                 "betweenness_centrality": round(betweenness.get(word, 0.0), 4),
#                 "frequency":              word_freq.get(word, 0),
#                 "polarity":               0.0,
#                 "topic_id":               -1,
#             })

#     if not rows:
#         return _generate_dummy_sna()

#     return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════
# BERTopic 신규 산출물 로더 (2026-05-10 도착)
# ═════════════════════════════════════════════════════════════
_BERT_SCOPE_PATHS = {
    "all":  "bert_110m",   # 전체 1.11M건
    "22m":  "bert_22m",    # 균형 213,972건
    "low":  "bert_low",    # 저평점 9,448건
}


@st.cache_data(ttl=CACHE_TTL, show_spinner="BERTopic 데이터 로드 중...")
def get_dashboard_reviews(scope: str = "22m") -> pd.DataFrame:
    """3_BERTopic.py 전용 — 새 dashboard_reviews_*.parquet 로더.

    Args:
        scope:
            "22m" — 균형 샘플 213,972건 (기본, 빠른 로드)
            "all" — 전체 1.11M건
            "low" — 저평점 9,448건 (별도 topic_low/topic_name_low/aspect_low 컬럼)

    Returns: brand, rating, topic, topic_name, aspect, aspect_label, content
        - aspect_label: BERT_ASPECT_KR 매핑 후 한글 표시명 (예: 핏/사이즈)
        - low scope는 컬럼명을 topic/topic_name/aspect로 통일하여 반환
        ※ review_id 부재 — ABSA join 불가, 페이지 단독 분석용
    """
    key = _BERT_SCOPE_PATHS.get(scope, "bert_22m")
    path = PATHS.get(key)
    if path is None or not path.exists():
        logger.info(f"[BERTopic] {key} 미존재 → 빈 DF 반환")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if scope == "low":
        # 컬럼명 통일 — topic_low → topic, topic_name_low → topic_name, aspect_low → aspect
        df = df.rename(columns={
            "topic_low": "topic",
            "topic_name_low": "topic_name",
            "aspect_low": "aspect",
        })

    # aspect 한글 매핑 — topic_name 대괄호 텍스트 폴백
    if "aspect" in df.columns:
        df["aspect_label"] = df["aspect"].map(BERT_ASPECT_KR)
    else:
        df["aspect_label"] = None
    _bracket = df["topic_name"].astype(str).str.extract(r"\[([^\]]+)\]")[0]
    df["aspect_label"] = df["aspect_label"].fillna(_bracket).fillna("기타")
    return df


@st.cache_data(ttl=CACHE_TTL)
def get_topic_dictionary(scope: str = "all") -> pd.DataFrame:
    """topic_aspect_mapping.parquet 로더 (49토픽 / 저평점 30토픽).

    Returns 원본 컬럼 + aspect_label(한글), keywords_top5(topic_keywords의 첫 5개).
    파일 부재 시 빈 DF.
    """
    key = "topic_map_low" if scope == "low" else "topic_map"
    path = PATHS.get(key)
    if path is None or not path.exists():
        return pd.DataFrame()
    td = pd.read_parquet(path)
    if "aspect" in td.columns:
        td["aspect_label"] = td["aspect"].map(BERT_ASPECT_KR).fillna(td["aspect"])
    if "topic_keywords" in td.columns and "keywords_top5" not in td.columns:
        def _top5(s):
            if not isinstance(s, str):
                return ""
            parts = [p.strip() for p in s.split(",") if p.strip()]
            return ", ".join(parts[:5])
        td["keywords_top5"] = td["topic_keywords"].apply(_top5)
    return td


# ═════════════════════════════════════════════════════════════
# 2. 파생 집계 (페이지 공용)
# ═════════════════════════════════════════════════════════════
@st.cache_data(ttl=CACHE_TTL)
def compute_brand_kpis(filters_hash: str = "") -> pd.DataFrame:
    """브랜드별 핵심 KPI — 시장 현황 페이지용.

    filters_hash 는 session 필터 변경 감지용 캐시 무효화 키.
    """
    reviews = get_reviews(columns=("review_id", "brand", "rating", "year"))
    if reviews.empty:
        return pd.DataFrame()

    grp = reviews.groupby("brand", observed=True)
    kpi = grp.agg(
        n_reviews=("review_id", "count"),
        mean_rating=("rating", "mean"),
        rating_std=("rating", "std"),
    ).reset_index()
    kpi["mean_rating"] = kpi["mean_rating"].round(2)
    kpi["rating_std"]  = kpi["rating_std"].round(2)
    return kpi.reindex([kpi.index[kpi["brand"] == b][0]
                        for b in BRAND_ORDER if b in kpi["brand"].values]).reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL)
def compute_aspect_polarity(filters_hash: str = "", sample_mode: str = "all") -> pd.DataFrame:
    """브랜드 × 6속성 P/N/X 비율 — ABSA 페이지용.

    Args:
        sample_mode: "all"=비대칭 전체(20K+3K×3), "balanced"=phase_e 균형(각 3,014)

    Returns: long-format [brand, aspect, P_ratio, N_ratio, X_ratio, n_reviews]
    """
    reviews = get_reviews(columns=("review_id", "brand"))
    absa    = get_absa(sample_mode=sample_mode)
    if absa.empty or reviews.empty:
        return pd.DataFrame()

    absa_clean = absa.drop(columns=[c for c in absa.columns if c in reviews.columns and c != "review_id"])
    df = reviews.merge(absa_clean, on="review_id", how="inner")
    rows = []
    for brand, sub in df.groupby("brand", observed=True):
        n = len(sub)
        if n < MIN_REVIEWS_FOR_BRAND_SCORE:
            continue
        for aspect in ASPECT_KEYS:
            counts = sub[aspect].value_counts(normalize=True)
            rows.append({
                "brand":  brand,
                "aspect": aspect,
                "P_ratio": float(counts.get("P", 0.0)),
                "N_ratio": float(counts.get("N", 0.0)),
                "X_ratio": float(counts.get("X", 0.0)),
                "n_reviews": n,
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=CACHE_TTL)
def compute_positioning_from_absa() -> pd.DataFrame:
    """ABSA → 브랜드 좌표 즉석 산출 (positioning_scores 부재 시 폴백).

    좌표 산출 공식:
      polarity_score = (P_ratio - N_ratio) / (P_ratio + N_ratio + ε)   ∈ [-1, 1]
      x_function     = 0.5 * (1 + functionality_score)                 ∈ [0, 1]
      y_heritage     = 0.5 * (1 + brand_heritage_score)                ∈ [0, 1]

    설계 노트:
    - minmax 정규화는 4브랜드 비교에서 항상 한 브랜드를 0, 한 브랜드를 1로
      찍는 부작용이 있어, 절대 스케일 (P-N)/(P+N) → [0,1] 선형 변환을
      사용한다. 0.5 = 중립(P=N), 1.0 = 전 긍정, 0.0 = 전 부정.
    - 해당 속성에서 ABSA 결과가 없는 브랜드는 NaN을 유지한다(0으로 채우면
      좌하단으로 잘못 시각화됨).
    """
    pol = compute_aspect_polarity()
    if pol.empty:
        return pd.DataFrame()

    func  = pol[pol["aspect"] == "functionality"].set_index("brand")
    brand = pol[pol["aspect"] == "brand_heritage"].set_index("brand")

    out = pd.DataFrame(index=BRAND_ORDER)
    eps = 1e-9
    fn_score = (func["P_ratio"] - func["N_ratio"]) / (func["P_ratio"] + func["N_ratio"] + eps)
    bh_score = (brand["P_ratio"] - brand["N_ratio"]) / (brand["P_ratio"] + brand["N_ratio"] + eps)

    # 절대 스케일 [-1,1] → [0,1] 선형 변환 (minmax 부작용 회피)
    out["x_function"] = 0.5 * (1 + fn_score)
    out["y_heritage"] = 0.5 * (1 + bh_score)

    # 신뢰구간 ±0.05 (NaN은 NaN으로 자연 전파)
    out["x_function_ci_low"]  = out["x_function"] - 0.05
    out["x_function_ci_high"] = out["x_function"] + 0.05
    out["y_heritage_ci_low"]  = out["y_heritage"] - 0.05
    out["y_heritage_ci_high"] = out["y_heritage"] + 0.05

    out["n_reviews"]   = func["n_reviews"]
    out["mean_rating"] = np.nan
    out["top_strengths"]  = [[] for _ in range(len(out))]
    out["top_weaknesses"] = [[] for _ in range(len(out))]
    out["top_topics"]     = [[] for _ in range(len(out))]
    out = out.reset_index().rename(columns={"index": "brand"})
    return out


# ═════════════════════════════════════════════════════════════
# 3. 더미 생성기 — 결정론적, 스키마 호환
# ═════════════════════════════════════════════════════════════
def _load_or_dummy(path: Path, schema: dict, name: str, dummy_fn) -> pd.DataFrame:
    if path.exists():
        df = pd.read_parquet(path)
        result = check_schema(df, schema, name)
        if result.missing:
            logger.warning(result.summary())
            warn_using_dummy(f"{name} (스키마 불일치 — 누락 컬럼: {result.missing})")
            return dummy_fn()
        return df
    logger.info(f"[DUMMY] {path.name} 미발견 → 더미 생성")
    return dummy_fn()


def _generate_dummy_reviews(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(DUMMY_SEED)
    cats = ["상의", "하의", "세트상품", "신발", "양말", "아우터"]
    sizes = ["S", "M", "L", "XL"]
    genders = ["women", "men", "unisex"]

    df = pd.DataFrame({
        "review_id": [f"R{i:07d}" for i in range(n)],
        "brand":     rng.choice(BRAND_ORDER, n, p=[0.15, 0.30, 0.35, 0.20]),
        "cat1":      rng.choice(cats, n),
        "cat2":      rng.choice(sizes, n),
        "cat3":      "",
        "gender":    rng.choice(genders, n),
        "rating":    pd.array(rng.choice([1, 2, 3, 4, 5], n, p=[0.03, 0.04, 0.08, 0.25, 0.60]), dtype="Int8"),
        "review_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 800, n), unit="D"),
    })
    df["year"]  = df["review_date"].dt.year.astype("Int16")
    df["month"] = df["review_date"].dt.month.astype("Int8")
    df["content"]       = "[더미] 리뷰 본문"
    df["content_clean"] = "[더미] 정제 본문"
    df["content_len"]   = pd.array(rng.integers(10, 200, n), dtype="Int32")
    df["tokens"]        = [["더미","토큰"] for _ in range(n)]
    df["tokens_topic"]  = [["더미","토픽"] for _ in range(n)]
    return df


def _generate_dummy_topics() -> pd.DataFrame:
    """리뷰별 토픽 할당 더미. reviews 의 review_id 와 정합성 유지."""
    rev = get_reviews(columns=("review_id",))
    if rev.empty:
        return pd.DataFrame(columns=list(TOPICS_SCHEMA.keys()))

    rng = np.random.default_rng(DUMMY_SEED + 1)
    topic_pool = [
        ("쿠셔닝/착용감", ["쿠션", "폭신", "푹신", "발편안", "착용감", "안정감", "충격흡수", "발걸음"]),
        ("핏/보정",        ["핏", "사이즈", "라인", "보정", "허리", "Y존", "압박감", "라이즈"]),
        ("디자인/색감",   ["디자인", "예쁘다", "색상", "이쁘다", "패턴", "컬러", "실물", "포인트"]),
        ("소재/촉감",     ["소재", "촉감", "원단", "부드럽다", "쫀쫀", "두께", "신축성", "기모"]),
        ("재구매/추천",   ["재구매", "추천", "또 살게요", "단골", "믿고", "팬", "다시", "여러개"]),
        ("가성비/할인",   ["가격", "가성비", "할인", "세일", "저렴", "비싸다", "혜택", "쿠폰"]),
        ("배송/포장",     ["배송", "포장", "빠르다", "꼼꼼", "박스", "도착", "발송", "구김"]),
        ("운동/활동성",   ["운동", "요가", "필라테스", "골프", "러닝", "활동성", "땀", "통기성"]),
    ]
    n = len(rev)
    tids = rng.integers(0, len(topic_pool), n)
    df = pd.DataFrame({
        "review_id": rev["review_id"].values,
        "topic_id":  pd.array(tids, dtype="Int16"),
        "topic_name": [topic_pool[t][0] for t in tids],
        "topic_label_auto": [f"topic_{t}" for t in tids],
        "topic_keywords":   [topic_pool[t][1] for t in tids],
        "probability":      rng.uniform(0.3, 0.95, n).astype("float32"),
    })
    return df


def _generate_dummy_absa() -> pd.DataFrame:
    rev = get_reviews(columns=("review_id", "brand", "rating"))
    if rev.empty:
        return pd.DataFrame(columns=list(ABSA_SCHEMA.keys()))

    rng = np.random.default_rng(DUMMY_SEED + 2)
    n = len(rev)

    # 브랜드별 사전 — 휠라(헤리티지 약/기능 중), 룰루레몬(둘 다 강) 시뮬
    brand_priors = {
        "FILA":     {"functionality": [0.30, 0.10, 0.60], "brand_heritage": [0.25, 0.15, 0.60]},
        "안다르":   {"functionality": [0.45, 0.10, 0.45], "brand_heritage": [0.35, 0.10, 0.55]},
        "젝시믹스": {"functionality": [0.50, 0.08, 0.42], "brand_heritage": [0.40, 0.08, 0.52]},
        "룰루레몬": {"functionality": [0.55, 0.12, 0.33], "brand_heritage": [0.55, 0.05, 0.40]},
    }
    default_prior = [0.40, 0.10, 0.50]

    out = {"review_id": rev["review_id"].values}
    for aspect in ASPECT_KEYS:
        labels = []
        for b in rev["brand"].values:
            if aspect in ("functionality", "brand_heritage"):
                p = brand_priors.get(str(b), {}).get(aspect, default_prior)
            else:
                p = default_prior
            labels.append(rng.choice(SENTIMENT_LABELS, p=p))
        out[aspect] = pd.Categorical(labels, categories=SENTIMENT_LABELS)
        out[f"{aspect}_confidence"] = rng.uniform(0.55, 0.98, n).astype("float32")
    return pd.DataFrame(out)


# ═════════════════════════════════════════════════════════════
# 4. 보조
# ═════════════════════════════════════════════════════════════
def _axis_hint_for_topic(name: str) -> str:
    if any(kw in name for kw in ["쿠셔닝", "기능", "활동", "운동", "통기"]):
        return "function"
    if any(kw in name for kw in ["재구매", "추천", "헤리티지", "브랜드"]):
        return "heritage"
    return "neutral"


def filters_to_hash(filters: dict) -> str:
    """session_state 의 필터 dict 를 안정적인 해시로 — 캐시 키."""
    s = "|".join(f"{k}={sorted(v) if isinstance(v, list) else v}" for k, v in sorted(filters.items()))
    return hashlib.md5(s.encode()).hexdigest()


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """공통 필터 적용기 — Graceful: 컬럼 없으면 무시."""
    if df.empty:
        return df
    out = df

    # 브랜드
    brands = filters.get("brands", [])
    if brands and "brand" in out.columns:
        out = out[out["brand"].isin(brands)]

    # 평점 (selectbox 단일값: "전체" / "1점" … "5점")
    rating_sel = filters.get("rating_sel", "전체")
    if rating_sel != "전체" and "rating" in out.columns:
        try:
            rating_val = int(rating_sel[0])
            out = out[out["rating"] == rating_val]
        except (ValueError, IndexError):
            pass

    # 연도 범위
    if "year_range" in filters and "year" in out.columns:
        lo, hi = filters["year_range"]
        out = out[(out["year"] >= lo) & (out["year"] <= hi)]

    # 카테고리1 / 2 / 3
    for col, key in [("cat1", "cat1_filters"), ("cat2", "cat2_filters"), ("cat3", "cat3_filters")]:
        vals = filters.get(key, [])
        if vals and col in out.columns:
            out = out[out[col].isin(vals)]

    # 가격 범위 (discount_price)
    if "price_range" in filters and "discount_price" in out.columns:
        lo, hi = filters["price_range"]
        out = out[(out["discount_price"] >= lo) & (out["discount_price"] <= hi)]

    return out
