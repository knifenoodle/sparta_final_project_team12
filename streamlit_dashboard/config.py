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

# ─────────────────────────────────────────────────────────────
# 1. 경로 설정 (로컬 & 배포 환경 하이브리드)
# ─────────────────────────────────────────────────────────────
APP_ROOT = Path(__file__).resolve().parent
# 배포 서버 환경(mount/src/...)에서는 APP_ROOT 하위의 data 폴더를 우선 사용합니다.
DATA_DIR = APP_ROOT / "data"

# 로컬 테스트용 절대 경로 (수정하신 기존 경로 유지 가능)
LOCAL_DIR = APP_ROOT.parent.parent / "송원우" / "final_data"
if LOCAL_DIR.exists():
    DATA_DIR = LOCAL_DIR

PATHS = {
    "absa":          DATA_DIR / "absa_phase_e_predictions.parquet",
    "bert_110m":     DATA_DIR / "dashboard_reviews_110M.parquet",
    "bert_22m":      DATA_DIR / "dashboard_reviews_22M.parquet",
    "bert_low":      DATA_DIR / "dashboard_reviews_low.parquet",
    "reviews":       DATA_DIR / "preprocessed_absa.parquet",
    "topic_map":     DATA_DIR / "topic_aspect_mapping.parquet",
    "topic_map_low": DATA_DIR / "low_topic_aspect_mapping.parquet",
}

# ─────────────────────────────────────────────────────────────
# 2. 한글 폰트 설정 (폰트 깨짐 해결)
# ─────────────────────────────────────────────────────────────
def setup_korean_font():
    """Streamlit Cloud 환경에서 한글 폰트를 설치하고 설정합니다."""
    try:
        # 리눅스 서버용 나눔 폰트 설치 명령어 (최초 1회 실행 권장)
        # 배포 시 패키지 매니저(packages.txt) 사용이 정석이나, 코드 내 방어 로직 추가
        if os.name != 'nt': # 리눅스(서버) 환경일 때
            font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            if not os.path.exists(font_path):
                os.system('sudo apt-get install -y fonts-nanum')
            
            # 폰트 적용
            fm.fontManager.addfont(font_path)
            plt.rc('font', family='NanumGothic')
    except Exception as e:
        st.warning(f"폰트 설정 중 오류 발생: {e}. 기본 폰트로 렌더링합니다.")

# ─────────────────────────────────────────────────────────────
# 3. 데이터 존재 확인 및 자동 다운로드 (이전 gdown 로직 유지)
# ─────────────────────────────────────────────────────────────
def ensure_data_exists():
    import gdown
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 앞서 정리한 GDRIVE_DATA 딕셔너리를 활용해 다운로드 수행
    # (생략: 이전 답변의 GDRIVE_DATA 딕셔너리 및 다운로드 루프 삽입)
def ensure_data_exists():
    """
    파일이 지정된 경로(DATA_DIR)에 없으면 구글 드라이브에서 자동으로 다운로드합니다.
    """
    import gdown
    
    # 배포 환경에서 데이터 폴더가 없으면 생성
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
    for key, info in GDRIVE_DATA.items():
        target_path = PATHS[key]
        if not target_path.exists():
            with st.spinner(f"데이터 파일 다운로드 중: {info['name']}... (약 450MB 전체 용량 중 일부)"):
                url = f"https://drive.google.com/uc?export=download&id={info['id']}"
                # 로컬에 파일이 없을 때만 구글 드라이브에서 다운로드 수행
                gdown.download(url, str(target_path), quiet=True)

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
