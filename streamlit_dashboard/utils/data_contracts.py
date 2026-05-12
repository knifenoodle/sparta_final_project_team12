"""
data_contracts.py — 모델팀과 합의할 Parquet 스키마 정의
========================================================

각 산출물(토픽/ABSA/포지셔닝/SNA)의 컬럼 명·타입·제약을 명시한 단일 계약서.
이 파일을 모델팀에 공유 → 어긋난 컬럼/타입은 data_loader 의 검증 단계에서
즉시 경고하여 사일로 간 통합 오류를 방지.

원칙:
- 각 테이블은 review_id 를 외래키로 가진다 (조인의 단일 키).
- 카테고리 필드는 string. 점수는 float32. 라벨은 P/N/X 카테고리.
- 결측 허용 컬럼은 NULLABLE 표기.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
# 1. reviews_master  ← 이미 존재 (preprocessed_absa.parquet)
# ─────────────────────────────────────────────────────────────
REVIEWS_SCHEMA = {
    "review_id":     "float64",     # PK (숫자형 ID, 실 parquet 기준)
    "brand":         "category",    # FILA / 안다르 / 젝시믹스 / 룰루레몬
    "cat1":          "string",      # 상의 / 하의 / 신발 ...
    "cat2":          "string",
    "cat3":          "string",      # NULLABLE
    "gender":        "category",    # women / men / unisex / kids
    "rating":        "int64",       # 1~5
    "review_date":   "string",      # YYYY-MM-DD 문자열
    "year":          "int64",
    "month":         "int64",
    "content":       "string",
    "content_clean": "string",
    "content_len":   "int64",
    "tokens":        "object",      # str (공백 구분 토큰)
    "tokens_topic":  "object",      # str (BERTopic 입력용)
}

# ─────────────────────────────────────────────────────────────
# 2. topic_results  (BERTopic 산출)
# ─────────────────────────────────────────────────────────────
TOPICS_SCHEMA = {
    "review_id":       "string",     # FK → reviews
    "topic_id":        "Int16",      # -1 = outlier
    "topic_name":      "string",     # 사람이 읽는 이름 (예: "쿠셔닝/착용감")
    "topic_label_auto": "string",    # BERTopic 자동 라벨
    "topic_keywords":  "object",     # list[str], 상위 c-TF-IDF 8개
    "probability":     "float32",    # 0~1
}

# 토픽 메타 (review-level과 별도, topic_id별 1행)
TOPIC_META_SCHEMA = {
    "topic_id":     "Int16",
    "topic_name":   "string",
    "n_reviews":    "Int32",
    "keywords":     "object",
    "axis_hint":    "string",        # "function" / "heritage" / "neutral"
    "representative_doc": "string",  # 대표 리뷰 1건
}

# ─────────────────────────────────────────────────────────────
# 3. absa_results  (EXAONE 캐스케이드 산출)
# ─────────────────────────────────────────────────────────────
ABSA_SCHEMA = {
    "review_id":             "string",   # FK
    "fit_size":              "category", # P / N / X
    "material_durability":   "category",
    "functionality":         "category",
    "design":                "category",
    "brand_heritage":        "category",
    "price_value":           "category",
    # 선택적 신뢰도(향후 확장)
    "fit_size_confidence":   "float32",  # NULLABLE
    "material_durability_confidence": "float32",
    "functionality_confidence":       "float32",
    "design_confidence":              "float32",
    "brand_heritage_confidence":      "float32",
    "price_value_confidence":         "float32",
}

# ─────────────────────────────────────────────────────────────
# 4. positioning_scores  (브랜드 단위 집계 — 최종 산출)
# ─────────────────────────────────────────────────────────────
POSITIONING_SCHEMA = {
    "brand":          "category",
    "x_function":     "float32",      # 0~1, 정규화된 기능성 강도
    "y_heritage":     "float32",      # 0~1, 정규화된 헤리티지 강도
    "x_function_ci_low":  "float32",
    "x_function_ci_high": "float32",
    "y_heritage_ci_low":  "float32",
    "y_heritage_ci_high": "float32",
    "n_reviews":      "Int32",
    "mean_rating":    "float32",
    "top_strengths":  "object",       # list[str], P 비율 상위 속성
    "top_weaknesses": "object",       # list[str], N 비율 상위 속성
    "top_topics":     "object",       # list[str], 대표 토픽
}

# ─────────────────────────────────────────────────────────────
# 5. sna_centrality  (NetworkX 키워드 그래프)
# ─────────────────────────────────────────────────────────────
SNA_SCHEMA = {
    "keyword":    "string",
    "brand":      "category",
    "centrality": "float32",       # eigenvector centrality
    "topic_id":   "Int16",
    "frequency":  "Int32",
    "polarity":   "float32",       # -1.0 ~ 1.0 (P비율 - N비율)
}


# ─────────────────────────────────────────────────────────────
# 검증 유틸
# ─────────────────────────────────────────────────────────────
@dataclass
class SchemaCheckResult:
    name: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    extra: list[str]   = field(default_factory=list)
    type_mismatch: dict[str, tuple[str, str]] = field(default_factory=dict)

    def summary(self) -> str:
        if self.ok:
            return f"✅ {self.name}: 스키마 일치"
        parts = [f"⚠️ {self.name} 스키마 이슈"]
        if self.missing:
            parts.append(f"  • 누락: {self.missing}")
        if self.extra:
            parts.append(f"  • 추가(무시 가능): {self.extra}")
        if self.type_mismatch:
            parts.append(f"  • 타입 불일치: {self.type_mismatch}")
        return "\n".join(parts)


def check_schema(df, schema: dict, name: str) -> SchemaCheckResult:
    """DataFrame 컬럼 ↔ schema 비교. 부드러운 검증(경고 수준)."""
    have = set(df.columns)
    want = set(schema.keys())

    missing = sorted(want - have)
    extra   = sorted(have - want)

    type_mismatch = {}
    for col, want_t in schema.items():
        if col not in df.columns:
            continue
        got_t = str(df[col].dtype)
        if want_t == got_t:
            continue
        # 느슨한 비교
        _str_like = ("object", "string", "str", "category")
        if want_t in ("string", "object") and got_t in _str_like:
            continue
        if want_t == "category" and got_t in _str_like:
            continue
        if want_t.lower().startswith("int") and got_t.lower().startswith("int"):
            continue
        if want_t.lower().startswith("float") and got_t.lower().startswith("float"):
            continue
        # float64 ↔ int64 (nullable int 포함) 허용
        if want_t in ("float64", "float32") and got_t.lower().startswith("int"):
            continue
        type_mismatch[col] = (want_t, got_t)

    ok = not missing and not type_mismatch
    return SchemaCheckResult(name, ok, missing, extra, type_mismatch)
