"""
exceptions.py — Graceful Degradation
=====================================

데이터 결측·필터 결과 0건·모델 산출물 미존재 등의 상황에서
대시보드가 빈 화면 또는 트레이스백을 노출하지 않도록 통일된
폴백 UI를 제공.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st


def empty_state(
    title: str,
    description: str = "",
    suggestion: str = "사이드바 필터를 완화하거나 모델팀 산출물 도착 후 새로고침하세요.",
    icon: str = "📭",
) -> None:
    """결과 0건 또는 데이터 부재 시 표시할 우아한 빈 상태 카드."""
    st.info(
        f"### {icon} {title}\n\n"
        f"{description}\n\n"
        f"**다음 행동:** {suggestion}"
    )


def warn_using_dummy(name: str) -> None:
    """더미 모드 사용 중임을 명시 (배너 형태)."""
    st.warning(
        f"⚙️ **{name}** 모델 산출물이 아직 도착하지 않아 더미 데이터로 표시 중입니다. "
        f"실제 Parquet 파일이 들어오면 자동 전환됩니다."
    )


@contextmanager
def safe_block(label: str = "이 영역"):
    """try/except 컨텍스트 매니저 — 컴포넌트 단위 격리."""
    try:
        yield
    except Exception as e:           # noqa: BLE001
        st.error(f"❌ {label} 렌더링 실패: `{type(e).__name__}: {e}`")
        with st.expander("디버그 정보"):
            st.exception(e)


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> bool:
    """컬럼 검증. 누락 시 에러 카드 + False 반환."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"⚠️ **{name}** 데이터에 컬럼 누락: `{missing}` — 모델팀 산출물 스키마 확인 필요.")
        return False
    return True


def check_min_rows(df: pd.DataFrame, n: int, name: str = "") -> bool:
    """최소 행수 검증. 미달 시 빈 상태 카드."""
    if len(df) < n:
        empty_state(
            title="표본 부족",
            description=f"{name} 데이터가 {len(df)}건으로, 안정적인 통계 산출 최소치({n}건) 미달입니다.",
        )
        return False
    return True


def data_health_check(paths: dict[str, Path]) -> pd.DataFrame:
    """홈/사이드바에 표시할 데이터 산출물 헬스 테이블."""
    rows = []
    for name, p in paths.items():
        rows.append({
            "산출물":   name,
            "경로":     p.name,
            "존재":     "✅" if p.exists() else "🟡 더미",
            "크기(MB)": f"{p.stat().st_size / 1e6:.1f}" if p.exists() else "—",
        })
    return pd.DataFrame(rows)
