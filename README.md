# Panel Insight Backend

https://github.com/user-attachments/assets/41e0dcd2-aeff-4c9c-8d2d-b4db43ce297c

Panel Insight은 패널 데이터 분석 및 클러스터링을 위한 백엔드 API 서버입니다.

## Preview
<img width="1152" height="624" alt="패널인사이트" src="https://github.com/user-attachments/assets/758fcc30-4a6f-4612-aebc-1835bcb5b857" />
<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/f0db4b7f-c47e-499f-982c-c8bd51325c93" width="500"/>
      <div style="font-size:12px;">패널 검색 결과 화면</div>
    </td>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/fc2bea71-7e88-4c2d-b70c-9e377b54f837" width="500"/>
      <div style="font-size:12px;">패널 상세 정보 및 인사이트</div>
    </td>
  </tr>
</table>

<table>
  <tr>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/d449dd77-7991-434d-8c3b-5e78c98c6ff8" width="500"/>
      <div style="font-size:12px;">클러스터링 UMAP 시각화 화면</div>
    </td>
    <td align="center" width="50%">
      <img src="https://github.com/user-attachments/assets/b74b8ebd-f018-4bf8-800e-11e8c62e969c" width="500"/>
      <div style="font-size:12px;">클러스터 비교 분석 화면</div>
    </td>
  </tr>
</table>


## Member
- 유성재 / Lead, AI, Data
- 김종유 / BE, FE
- 문재원 / BE, AI
- 김민수 / Data


## Tech Stack

- **FastAPI 0.115.0** - 웹 프레임워크
- **Python 3.13+** - 런타임
- **PostgreSQL (NeonDB)** - 데이터베이스
- **pgvector** - 벡터 검색 확장
- **SQLAlchemy 2.0** - 비동기 ORM
- **HDBSCAN** - 밀도 기반 클러스터링
- **UMAP** - 차원 축소 및 시각화
- **Pandas** - 데이터 처리
- **NumPy** - 수치 연산
- **scikit-learn** - 머신러닝 유틸리티
- **Pinecone** - 벡터 검색 엔진
- **OpenAI API** - 텍스트 임베딩 생성
- **Anthropic Claude API** - 메타데이터 추출 및 카테고리 분류

## Getting Started

### Installation

```bash
git clone https://github.com/hansung-sw-capstone-2025-2/2025_8_G_BE.git
cd 2025_8_G_BE
```

```bash
cd server
pip install -r requirements.txt
```

### Environment Variables

`.env` 파일을 생성하고 다음 변수를 설정하세요:

```env
# Database
DATABASE_URL=postgresql://user:password@host:port/database

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=panel-profiles
PINECONE_ENVIRONMENT=us-east-1

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Logging
LOG_LEVEL=INFO
```

### Development

```bash
cd server
python run_server.py
```

또는 uvicorn을 직접 사용:

```bash
cd server
uvicorn app.main:app --reload --host 127.0.0.1 --port 8004
```

### Build

프로덕션 빌드는 별도로 필요하지 않습니다. Python 애플리케이션이므로 직접 실행합니다.

## Project Structure

```
server/
├── app/
│   ├── main.py              # FastAPI 앱 진입점
│   ├── api/                 # API 엔드포인트
│   │   ├── search.py        # 벡터 검색 API
│   │   ├── panels.py        # 패널 관련 API
│   │   ├── pinecone_panel_detail.py # Pinecone 패널 상세 정보
│   │   ├── pinecone_panel_details.py # Pinecone 패널 상세 정보
│   │   ├── clustering.py   # 클러스터링 API
│   │   ├── clustering_viz.py # 클러스터링 시각화 API
│   │   ├── precomputed.py   # 사전 계산된 데이터 API
│   │   └── health.py        # Health check API
│   ├── clustering/          # 클러스터링 로직
│   │   ├── algorithms/      # 클러스터링 알고리즘
│   │   │   ├── hdbscan.py
│   │   │   └── base.py
│   │   ├── core/            # 핵심 클러스터링 로직
│   │   │   ├── pipeline.py
│   │   │   └── strategy_manager.py
│   │   ├── filters/         # 필터 로직
│   │   └── processors/      # 데이터 처리기
│   ├── db/                  # 데이터베이스 관련
│   │   ├── session.py       # DB 세션 관리
│   │   ├── dao_panels.py    # 패널 데이터 액세스
│   │   └── dao_embeddings.py # 임베딩 데이터 액세스
│   ├── services/            # 비즈니스 로직
│   │   ├── pinecone_pipeline.py # Pinecone 검색 파이프라인
│   │   ├── metadata_extractor.py # 메타데이터 추출
│   │   ├── metadata_filter_extractor.py # 카테고리별 메타 데이터 필터 추출
│   │   ├── category_classifier.py # 카테고리 분류
│   │   ├── pinecone_filter_converter.py # Pinecone 필터 변환
│   │   ├── text_generator.py # 카테고리별 텍스트 생성
│   │   ├── embedding_generator.py # OpenAI 임베딩 생성
│   │   ├── pinecone_searcher.py # Pinecone 검색기
│   │   ├── pinecone_result_filter.py # Pinecone 결과 필터
│   │   └── lifestyle_classifier.py # ai 인사이트 생성
│   ├── core/                # 핵심 설정
│   │   └── config.py        # 설정 관리
│   └── utils/               # 유틸리티 함수
├── requirements.txt         # Python dependencies
└── run_server.py            # 서버 실행 스크립트
```

## Key Features

- **의미 기반 벡터 검색**: Pinecone과 OpenAI Embeddings를 활용한 임베딩 기반 검색
- **AI 인사이트**: 	Claude Opus 4.1 을 활용한 패널 ai 인사이트
- **HDBSCAN 클러스터링**: 밀도 기반 클러스터링 알고리즘
- **UMAP 시각화**: 고차원 클러스터링 결과를 2D 공간으로 시각화
- **클러스터 비교 분석**: 클러스터 간 통계적 비교 분석
- **사전 계산 데이터**: 성능 최적화를 위한 사전 계산된 클러스터 데이터
- **패널 데이터 관리**: NeonDB를 통한 패널 데이터 조회 및 관리

## API Endpoints

### Search API (`/api/search`)

- `POST /api/search` - 의미 기반 벡터 검색 (임베딩 기반)
- `GET /api/search/status` - 검색 상태 확인

### Panel API (`/api/panels`)

- `GET /api/panels/{panel_id}` - 패널 상세정보 조회
- `POST /api/panels/batch` - 여러 패널 상세정보 배치 조회

### Clustering API (`/api/clustering`)

- `POST /api/clustering/cluster-around-search` - 검색된 패널 주변 클러스터 분석
- `GET /api/clustering/umap` - UMAP 좌표 조회
- `GET /api/clustering/panel-cluster-mapping` - 패널-클러스터 매핑 조회

### Precomputed API (`/api/precomputed`)

- `GET /api/precomputed/clusters` - 사전 계산된 클러스터 정보 조회
- `GET /api/precomputed/umap` - UMAP 좌표 조회
- `GET /api/precomputed/comparison/{cluster_a}/{cluster_b}` - 클러스터 비교 데이터 조회
- `GET /api/precomputed/profiles` - 클러스터 프로필 조회

### Health Check (`/health`)

- `GET /health` - 기본 Health check
- `GET /health/db` - 데이터베이스 연결 상태 확인
- `GET /healthz` - 종합 Health check

## API Documentation

서버 실행 후 아래 URL에서 확인:

- **FastAPI Docs**: http://127.0.0.1:8004/docs
- **ReDoc**: http://127.0.0.1:8004/redoc


## LLM MODELS
- claude-opus-4-1: AI 인사이트 생성
- claude-haiku-4-5: 메타데이터 추출(쿼리 파싱), 카테고리 분류, 텍스트 생성


## License

이 프로젝트는 한성대학교 기업연계 SW캡스톤디자인 수업에서 진행되었습니다.
