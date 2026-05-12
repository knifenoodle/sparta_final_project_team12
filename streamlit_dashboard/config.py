"""
config.py — 대시보드 전역 설정
================================

브랜드 메타, 색상 팔레트, ABSA 속성, 임계값 등 모든 페이지·컴포넌트가
공유하는 상수의 단일 진실 공급원(Single Source of Truth).
"""
import os
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 1. 기초 경로 설정 (OS 독립적 경로 관리)
APP_ROOT = Path(__file__).resolve().parent
# Mac/Linux 호환을 위해 사용자 홈 디렉토리 기반 경로 설정 권장
LOCAL_DATA_DIR = Path.home() / "Documents" / "final_data" 
DEPLOY_DATA_DIR = APP_ROOT / "data"

# 환경에 따른 데이터 디렉토리 결정
DATA_DIR = LOCAL_DATA_DIR if LOCAL_DATA_DIR.exists() else DEPLOY_DATA_DIR

# 2. GDRIVE 데이터 정보
GDRIVE_DATA = {
    "reviews":       {"id": "1eSBnmXb6vaSWakPQWlnjdJItOrqeCZ_G", "name": "preprocessed_absa.parquet"},
    "absa":          {"id": "1UTKFCjepnuWPu5imLnThTyoZWe41tipo", "name": "absa_phase_e_predictions.parquet"},
    "bert_low":      {"id": "1AtMr8f6YV_J_i8aSo_MJ9pta3nsdnvq9", "name": "dashboard_reviews_low.parquet"},
    "bert_22m":      {"id": "10YsxXtiqbfev40RemeqyWFdkkBCoahw9", "name": "dashboard_reviews_22M.parquet"},
    "bert_110m":     {"id": "1uftIjOVOClHKiabsP3A0rswylSrOIf5p", "name": "dashboard_reviews_110M.parquet"},
    "topic_map":     {"id": "1njAnnqwT0crWl8NCFICZIl2rYS9CO2h1", "name": "topic_aspect_mapping.parquet"},
    "topic_map_low": {"id": "1mSmDz0yoFXpjA7YDJGUeomgXP7ExVM_S", "name": "low_topic_aspect_mapping.parquet"}
}

PATHS = {k: DATA_DIR / v["name"] for k, v in GDRIVE_DATA.items()}

# 3. 데이터 로드 및 환경 설정 (st.cache_resource 사용으로 속도 최적화)
@st.cache_resource
def initialize_environment():
    import gdown
    # 폴더 생성
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 데이터 다운로드
    for key, info in GDRIVE_DATA.items():
        target_path = PATHS[key]
        if not target_path.exists():
            url = f"https://drive.google.com/uc?export=download&id={info['id']}"
            gdown.download(url, str(target_path), quiet=True)
    
    # 폰트 설정
    linux_font = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(linux_font):
        fm.fontManager.addfont(linux_font)
        plt.rc('font', family='NanumGothic')
    else:
        # Mac/Windows 로컬 환경
        plt.rc('font', family='AppleGothic' if os.name == 'posix' else 'Malgun Gothic')
    
    return True

# ─────────────────────────────────────────────────────────────
# 브랜드 메타 — 자사/경쟁사 구분, 색상, 표시 이름
# ─────────────────────────────────────────────────────────────
BRANDS = {
    "FILA":   {"label": "휠라(FILA)",   "color": "#004B87", "is_self": True},
    "안다르": {"label": "안다르",       "color": "#D6CDC0", "is_self": False},
    "젝시믹스": {"label": "젝시믹스",   "color": "#1A1A1A", "is_self": False},
    "룰루레몬": {"label": "룰루레몬",   "color": "#C8102E", "is_self": False},
}
BRAND_ORDER = ["FILA", "안다르", "젝시믹스", "룰루레몬"]
BRAND_COLORS = {b: m["color"] for b, m in BRANDS.items()}

# ─────────────────────────────────────────────────────────────
# ABSA 6속성 — 한글 ↔ 영문 키 매핑 (ABSA EXAONE 추론 결과 기준)
# ─────────────────────────────────────────────────────────────
ASPECTS = [
    {"key": "fit_size",            "label": "핏/사이즈",       "axis": "x_supp"},
    {"key": "material_durability", "label": "소재/내구성",     "axis": "x_supp"},
    {"key": "functionality",       "label": "기능성",          "axis": "x_core"},
    {"key": "design",              "label": "디자인",          "axis": "y_supp"},
    {"key": "brand_heritage",      "label": "브랜드/헤리티지", "axis": "y_core"},
    {"key": "price_value",         "label": "가격/가치",       "axis": "supp"},
]
ASPECT_KEYS   = [a["key"] for a in ASPECTS]
ASPECT_LABELS = {a["key"]: a["label"] for a in ASPECTS}
LABEL_TO_KEY  = {a["label"]: a["key"] for a in ASPECTS}

# ─────────────────────────────────────────────────────────────
# BERTopic 산출 aspect (dashboard_reviews_*.parquet, topic_aspect_mapping.parquet)
# - ABSA 6속성과 키 체계가 다름. 별도 매핑 사전 사용.
# ─────────────────────────────────────────────────────────────
BERT_ASPECT_KR: dict[str, str] = {
    "aspect_size":     "핏/사이즈",
    "aspect_material": "소재/내구성",
    "aspect_quality":  "품질/내구성",
    "aspect_function": "기능성",
    "aspect_design":   "디자인",
    "aspect_brand":    "브랜드/헤리티지",
    "aspect_price":    "가격/가치",
    "aspect_other":    "기타",
    # 약어 호환
    "size":     "핏/사이즈",
    "material": "소재/내구성",
    "quality":  "품질/내구성",
    "function": "기능성",
    "design":   "디자인",
    "brand":    "브랜드/헤리티지",
    "price":    "가격/가치",
    "other":    "기타",
}

# ─────────────────────────────────────────────────────────────
# aspect → 컬러 매핑 (애슬레저 코어 : 네이비 & 레드 팔레트)
# ─────────────────────────────────────────────────────────────
BERT_ASPECT_COLOR: dict[str, str] = {
    "기능성":          "#660000",
    "브랜드/헤리티지": "#FF4D4D",  
    "품질/내구성":     "#CD5B5B",  
    "핏/사이즈":       "#00205B",  
    "디자인":          "#B0C4DE",  
    "소재/내구성":     "#6495ED",  
    "가격/가치":       "#AB6868",  
    "기타":            "#DBECF9", 
}

SENTIMENT_LABELS = ["P", "N", "X"]   # Positive / Negative / 없음
SENTIMENT_COLOR  = {"P": "#004B87", "N": "#C8102E", "X": "#9E9E9E"}

# ─────────────────────────────────────────────────────────────
# 시각화 임계값
# ─────────────────────────────────────────────────────────────
MIN_REVIEWS_FOR_BRAND_SCORE = 30   # 한 셀 미만 → graceful degradation
TOPIC_TOP_K_KEYWORDS = 8
POSITIONING_AXIS_RANGE = (0.0, 1.0)

# ─────────────────────────────────────────────────────────────
# 캐시 TTL (초) — 30분 기본
# ─────────────────────────────────────────────────────────────
CACHE_TTL = 1800 

# ─────────────────────────────────────────────────────────────
# Streamlit page 레이아웃 공통
# ─────────────────────────────────────────────────────────────
PAGE_LAYOUT = {
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

APP_TITLE = "FILA 애슬레저 시장 진입 전략 대시보드"
APP_SUBTITLE = "117만 건 리뷰 기반 — 기능성 × 헤리티지 포지셔닝"

# ─────────────────────────────────────────────────────────────
# 브랜드별 연도별 매출 (단위: 억 원)
# 룰루레몬 2025: 회계연도(2월~1월) 기준 성장흐름 유지 가정 추산치
# ─────────────────────────────────────────────────────────────
BRAND_SALES = {
    "FILA":   {"2023": 3_676, "2024": 3_668, "2025": 3_863, "est": False},
    "안다르": {"2023": 2_026, "2024": 2_368, "2025": 3_000, "est": False},
    "젝시믹스": {"2023": 2_166, "2024": 2_508, "2025": 2_503, "est": False},
    "룰루레몬": {"2023": 1_173, "2024": 1_567, "2025": 2_093, "est": True},
}
