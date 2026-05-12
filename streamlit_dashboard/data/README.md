# 데이터 디렉토리 안내

대시보드는 **`송원우/final_data/`** 의 Parquet 파일을 직접 참조합니다.
이 폴더(`송원우/streamlit_dashboard/data/`)는 더미·캐시·임시 출력용 보조 폴더입니다.

## 모델팀 인계 파일 (Data Contract)

| 파일 | 용도 | 스키마 |
|------|------|--------|
| `preprocessed_absa.parquet` | 리뷰 마스터 (1.16M) | `data_contracts.REVIEWS_SCHEMA` |
| `topic_results.parquet` | BERTopic 산출 | `TOPICS_SCHEMA` |
| `absa_predictions_full.parquet` | EXAONE ABSA 산출 | `ABSA_SCHEMA` |
| `positioning_scores.parquet` | 브랜드 좌표 (선택) | `POSITIONING_SCHEMA` |
| `sna_centrality.parquet` | NetworkX 중심성 | `SNA_SCHEMA` |

각 스키마의 컬럼명·타입은 `utils/data_contracts.py` 의 단일 정의를 따릅니다.
파일 미존재 시 `data_loader.py` 가 결정론적 더미를 생성하므로, 모델 작업과
대시보드 작업이 완전히 병렬로 진행될 수 있습니다.
