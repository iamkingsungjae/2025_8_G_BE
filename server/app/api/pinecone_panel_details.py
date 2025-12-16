"""Pinecone에서 패널 상세 정보 가져오기 (Pinecone 메타데이터만 사용)"""
from typing import List, Dict, Any
from datetime import datetime
import logging
import numpy as np

from app.services.pinecone_searcher import PineconePanelSearcher
from app.core.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, load_category_config
# ⭐ merged_data는 패널 상세정보 조회 시(/api/panels/{panel_id})에만 NeonDB에서 로드
# 검색 결과에서는 Pinecone 메타데이터만 사용
from pinecone import Pinecone

logger = logging.getLogger(__name__)


def _is_no_response(text: str) -> bool:
    """텍스트가 무응답인지 확인"""
    if not text:
        return True
    no_response_patterns = [
        "무응답", "응답하지 않았", "정보 없음", "해당 없음",
        "해당사항 없음", "기록 없음", "데이터 없음"
    ]
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in no_response_patterns)

# Pinecone 검색기 싱글톤
_pinecone_searcher: PineconePanelSearcher = None

# Pinecone 인덱스 싱글톤 (성능 최적화)
_pinecone_client = None
_pinecone_index = None
_pinecone_dimension = None

def _get_pinecone_searcher() -> PineconePanelSearcher:
    """Pinecone 검색기 싱글톤 인스턴스 반환"""
    global _pinecone_searcher
    
    if _pinecone_searcher is None:
        category_config = load_category_config()
        _pinecone_searcher = PineconePanelSearcher(
            PINECONE_API_KEY,
            PINECONE_INDEX_NAME,
            category_config
        )
    
    return _pinecone_searcher


async def _get_panel_details_from_pinecone(
    mb_sn_list: List[str],
    page: int,
    limit: int,
    similarity_scores: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    mb_sn 리스트로부터 패널 상세 정보 조회 (Pinecone 메타데이터만 사용)
    
    Args:
        mb_sn_list: 패널 ID 리스트
        page: 페이지 번호
        limit: 페이지당 항목 수
        
    Returns:
        기존 API 형식의 검색 결과
    """
    import time
    convert_start_time = time.time()
    logger.info(f"[Panel Details] 시작: {len(mb_sn_list)}개 패널, page={page}, limit={limit}")
    
    if not mb_sn_list:
        return {
            "results": [],
            "total": 0,
            "page": page,
            "page_size": limit,
            "pages": 0,
            "count": 0
        }
    
    category_config = load_category_config()
    
    # ⭐ Pinecone 인덱스 연결 최적화 (싱글톤 패턴 사용)
    # Pinecone 클라이언트는 재사용 가능하므로 매번 생성하지 않음
    global _pinecone_client, _pinecone_index, _pinecone_dimension
    
    if _pinecone_index is None:
        _pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = _pinecone_client.Index(PINECONE_INDEX_NAME)
        # 인덱스 차원 동적으로 가져오기 (한 번만)
        stats = _pinecone_index.describe_index_stats()
        _pinecone_dimension = stats.get('dimension', 1536)
    
    index = _pinecone_index
    dimension = _pinecone_dimension
    
    # Pinecone에서 인구 topic으로 메타데이터 조회
    # 랜덤 벡터로 검색 (메타데이터만 필요)
    random_vector = np.random.rand(dimension).astype(np.float32).tolist()
    norm = np.linalg.norm(random_vector)
    if norm > 0:
        random_vector = (np.array(random_vector) / norm).tolist()
    
    # 모든 topic에서 메타데이터 조회
    panel_metadata_map = {}  # mb_sn -> 모든 메타데이터 병합
    
    # 각 카테고리의 topic으로 메타데이터 수집
    logger.info(f"[Panel Details] 메타데이터 수집 시작: {len(mb_sn_list)}개 패널, {len(category_config)}개 카테고리")
    try:
        category_count = 0
        for category_name, category_info in category_config.items():
            category_count += 1
            pinecone_topic = category_info.get("pinecone_topic")
            if not pinecone_topic:
                continue
            
            try:
                topic_results = index.query(
                    vector=random_vector,
                    top_k=len(mb_sn_list) * 2,  # 충분히 많이 가져오기
                    include_metadata=True,
                    filter={
                        "topic": pinecone_topic,
                        "mb_sn": {"$in": mb_sn_list}
                    }
                )
                
                # mb_sn별로 메타데이터 병합
                for match in topic_results.matches:
                    mb_sn = match.metadata.get("mb_sn", "")
                    if not mb_sn:
                        continue
                    
                    if mb_sn not in panel_metadata_map:
                        panel_metadata_map[mb_sn] = {
                            "_index_values": []  # welcome1, welcome2, quickpoll 구분용
                        }
                    
                    # index 필드 수집 (welcome1, welcome2, quickpoll 구분용)
                    if "index" in match.metadata:
                        index_val = match.metadata.get("index")
                        if index_val and index_val not in panel_metadata_map[mb_sn]["_index_values"]:
                            panel_metadata_map[mb_sn]["_index_values"].append(index_val)
                    
                    # 메타데이터 병합 (시스템 필드 제외)
                    for key, value in match.metadata.items():
                        if key not in ["topic", "index", "mb_sn"]:  # 시스템 필드 제외
                            # 중복 키는 나중 값으로 덮어쓰기 (또는 리스트로 병합)
                            if key in panel_metadata_map[mb_sn]:
                                # 이미 있는 경우, 리스트면 병합, 아니면 덮어쓰기
                                existing = panel_metadata_map[mb_sn][key]
                                if isinstance(existing, list) and isinstance(value, list):
                                    panel_metadata_map[mb_sn][key] = list(set(existing + value))
                                elif isinstance(existing, list):
                                    if value not in existing:
                                        panel_metadata_map[mb_sn][key] = existing + [value]
                                else:
                                    panel_metadata_map[mb_sn][key] = value
                            else:
                                panel_metadata_map[mb_sn][key] = value
            except Exception as e:
                logger.warning(f"{category_name} ({pinecone_topic}) 메타데이터 조회 실패: {e}")
    except Exception as e:
        logger.warning(f"Pinecone 메타데이터 조회 실패: {e}, 빈 메타데이터 사용")
    
    # 결과 변환 (Pinecone 메타데이터만 사용)
    # ⭐ 성능 최적화: 검색 결과에서는 기본 메타데이터만 반환
    # 응답, AI 요약 등은 패널 상세 정보창에서만 로딩 (/api/panels/{panel_id})
    logger.info(f"[Panel Details] 메타데이터 수집 완료: {len(panel_metadata_map)}개 패널, 결과 변환 시작 (응답 제외)")
    
    # ⭐ 노트북 기반 유사도로 정렬 후 페이지네이션 적용
    # similarity_scores가 있으면 유사도 기준으로 정렬, 없으면 원래 순서 유지
    if similarity_scores:
        # 유사도 점수 기준으로 정렬 (내림차순 - 높은 점수부터)
        sorted_mb_sn_list = sorted(
            mb_sn_list,
            key=lambda mb_sn: similarity_scores.get(mb_sn, 0.0),
            reverse=True
        )
        logger.info(f"[Panel Details] 유사도 점수 기준으로 정렬 완료 (상위 5개 점수: {[similarity_scores.get(mb_sn, 0.0) for mb_sn in sorted_mb_sn_list[:5]]})")
    else:
        # 유사도 점수가 없으면 원래 순서 유지
        sorted_mb_sn_list = mb_sn_list
        logger.info(f"[Panel Details] 유사도 점수 없음, 원래 순서 유지")
    
    total_count = len(sorted_mb_sn_list)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    paginated_mb_sn_list = sorted_mb_sn_list[start_idx:end_idx]
    
    logger.info(f"[Panel Details] 페이지네이션: 전체 {total_count}개 중 {start_idx}~{end_idx}번째 ({len(paginated_mb_sn_list)}개) 처리")
    
    # ⭐ merged 데이터도 함께 로드하여 metadata에 병합 (SummaryBar 통계 계산을 위해)
    # 배치로 merged 데이터 조회
    from app.utils.merged_data_loader import get_panels_from_merged_db_batch
    merged_data_map = {}
    try:
        merged_data_map = await get_panels_from_merged_db_batch(paginated_mb_sn_list)
        logger.info(f"[Panel Details] merged 데이터 로드 완료: {len(merged_data_map)}개 패널")
    except Exception as e:
        logger.warning(f"[Panel Details] merged 데이터 로드 실패: {e}, Pinecone 메타데이터만 사용")
    
    # ⭐ 검색 결과에서는 Pinecone 메타데이터 + merged 데이터 병합
    results = []
    for mb_sn in paginated_mb_sn_list:
        metadata = panel_metadata_map.get(mb_sn, {})
        
        # ⭐ merged 데이터 가져오기 (fallback용)
        merged_data = merged_data_map.get(mb_sn, {})
        
        # 기본 정보 추출 (Pinecone 메타데이터 우선, 없으면 merged_data 사용)
        gender = metadata.get("성별", "") or merged_data.get("gender", "")
        region = metadata.get("지역", "") or merged_data.get("location", "")
        detail_location = metadata.get("지역구", "") or merged_data.get("detail_location", "")
        
        # 지역 정보 조합
        if detail_location and region:
            region = f"{region} {detail_location}"
        elif detail_location:
            region = detail_location
        elif not region and merged_data.get("location"):
            region = merged_data.get("location", "")
            if merged_data.get("detail_location"):
                region = f"{region} {merged_data.get('detail_location')}"
        
        age = 0
        # Pinecone 메타데이터에서 나이 가져오기
        if metadata.get("나이"):
            try:
                age = int(float(metadata["나이"]))
            except (ValueError, TypeError):
                pass
        
        # merged_data에서 나이 가져오기 (Pinecone에 없을 때)
        if not age and merged_data.get("age"):
            try:
                age = int(float(merged_data["age"]))
            except (ValueError, TypeError):
                pass
        
        # 연령대에서 나이 추정 (나이가 없을 때)
        if not age and metadata.get("연령대"):
            age_group = metadata["연령대"]
            if "대" in age_group:
                try:
                    age_base = int(age_group.replace("대", ""))
                    age = age_base + 5  # 중간값 사용
                except (ValueError, TypeError):
                    pass
        
        # 소득 정보
        income = ""
        if metadata.get("개인소득"):
            income = str(metadata["개인소득"])
        elif metadata.get("가구소득"):
            income = str(metadata["가구소득"])
        
        # ⭐ 성능 최적화: 검색 결과에서는 응답 수집하지 않음
        # 응답 정보는 패널 상세 정보창을 열 때만 로딩 (get_panel_from_pinecone 사용)
        responses = []  # 빈 리스트 (상세 정보는 별도 API에서 로딩)
        
        # AI 요약도 검색 결과에서는 제외 (상세 정보에서만 로딩)
        ai_summary = ""  # 빈 문자열 (상세 정보는 별도 API에서 로딩)
        
        # 유사도 점수 가져오기 (없으면 0.0)
        similarity = 0.0
        if similarity_scores and mb_sn in similarity_scores:
            similarity = similarity_scores[mb_sn]
        
        # 시스템 필드 제외하고 실제 데이터만 포함
        clean_metadata = {}
        for key, value in metadata.items():
            if key not in ["topic", "index", "mb_sn", "text", "_index_values"]:  # 시스템 필드 제외
                clean_metadata[key] = value
        
        # ⭐ merged 데이터를 metadata에 병합 (SummaryBar 통계 계산을 위해)
        # merged_data는 이미 위에서 가져왔음
        if merged_data:
            # merged_data의 base_profile 필드들을 metadata에 병합
            # 시스템 필드 및 중복 필드 제외 (Pinecone 메타데이터 우선)
            exclude_fields = ['mb_sn', 'quick_answers', 'id', 'name', 'gender', 'age', 'region', 'income', 'location', 'detail_location']
            
            # SummaryBar에 필요한 한글 필드명 목록 (모든 가능한 변형 포함)
            important_fields = [
                # 차량 관련
                '보유차량여부', '보유차량', 'car_ownership', 'car',
                # 휴대폰 관련
                '보유 휴대폰 단말기 브랜드', '휴대폰 브랜드', 'phone_brand', 'phone',
                # 흡연/음주 관련
                '흡연경험', '음용경험 술', '궐련형 전자담배/가열식 전자담배 이용경험',
                'smoking_experience', 'drinking_experience',
                # 가족 관련
                '자녀수', '가족수', '결혼여부', 'children_count', 'family_size', 'marriage_status',
                # 직업/소득 관련
                '직업', '직무', '월평균 개인소득', '월평균 가구소득', '개인소득', '가구소득',
                'occupation', 'job_role', 'personal_income', 'household_income',
                # 학력 관련
                '최종학력', 'education',
                # 기타
                '보유전제품'
            ]
            
            # 디버깅: merged_data의 키 확인 (처음 패널만)
            if mb_sn == paginated_mb_sn_list[0] if paginated_mb_sn_list else False:
                merged_keys = [k for k in merged_data.keys() if k not in exclude_fields]
                important_keys_found = [k for k in merged_keys if k in important_fields]
                logger.info(f"[Panel Details] merged_data 키 샘플 (mb_sn={mb_sn}): {merged_keys[:20]}")
                logger.info(f"[Panel Details] important_fields 매칭: {important_keys_found}")
            
            # ⭐ 모든 merged_data 필드를 clean_metadata에 병합 (SummaryBar 통계를 위해)
            for key, value in merged_data.items():
                if key not in exclude_fields and value is not None:
                    # SummaryBar에 필요한 필드들은 항상 merged 데이터로 덮어쓰기 (우선순위 최고)
                    # 다른 필드는 clean_metadata에 없을 때만 추가
                    if key in important_fields:
                        clean_metadata[key] = value
                    elif key not in clean_metadata:
                        clean_metadata[key] = value
        
        # ⭐ 검색 결과에서는 Pinecone 메타데이터 + merged 데이터 병합 완료
        original_job = ""
        original_job_role = ""
        
        # ⭐ income을 clean_metadata에서 다시 확인 (merged_data 병합 후)
        # 모든 가능한 소득 필드명 확인
        if not income:
            income = (clean_metadata.get("월평균 개인소득") or 
                     clean_metadata.get("개인소득") or 
                     clean_metadata.get("월평균 가구소득") or 
                     clean_metadata.get("가구소득") or
                     clean_metadata.get("personal_income") or
                     clean_metadata.get("household_income") or "")
            if income:
                income = str(income)
        
        # ⭐ coverage 계산 (QuickPoll 응답 여부 확인)
        coverage = None
        index_values = metadata.get("_index_values", [])
        has_w1 = "w1" in index_values or any("w1" in str(v) for v in index_values)
        has_w2 = "w2" in index_values or any("w2" in str(v) for v in index_values)
        has_q = "q" in index_values or any("q" in str(v) for v in index_values)
        
        # merged_data에서 quick_answers 확인 (Pinecone에 없을 때 fallback)
        # ⭐ 실제로 유효한 QuickPoll 응답이 있는지 확인
        if not has_q and merged_data:
            quick_answers = merged_data.get("quick_answers", {})
            if quick_answers and isinstance(quick_answers, dict):
                # 빈 딕셔너리가 아니고, 실제로 값이 있는 항목이 있는지 확인
                # quick_answers의 값이 None이 아니고, 빈 문자열이 아닌 항목이 하나라도 있어야 함
                has_valid_quick_answer = False
                for key, value in quick_answers.items():
                    if value is not None and value != "" and value != []:
                        # 리스트인 경우 비어있지 않은지 확인
                        if isinstance(value, list) and len(value) > 0:
                            has_valid_quick_answer = True
                            break
                        # 문자열이나 다른 타입인 경우 값이 있으면 유효
                        elif not isinstance(value, list):
                            has_valid_quick_answer = True
                            break
                if has_valid_quick_answer:
                    has_q = True
        
        if has_q and has_w1 and has_w2:
            coverage = "qw"  # Q + W1 + W2
        elif has_q and has_w1:
            coverage = "qw1"  # Q + W1
        elif has_q and has_w2:
            coverage = "qw2"  # Q + W2
        elif has_q:
            coverage = "q"  # Q만
        elif has_w1 and has_w2:
            coverage = "w"  # W1 + W2
        elif has_w1:
            coverage = "w1"  # W1만
        elif has_w2:
            coverage = "w2"  # W2만
        
        results.append({
            "id": mb_sn,
            "name": mb_sn,
            "mb_sn": mb_sn,
            "gender": gender,
            "age": age,
            "region": region,
            "income": str(income) if income else "",
            "coverage": coverage,  # ⭐ coverage 추가
            "welcome1_info": {
                "gender": gender,
                "age": age,
                "region": region,
                "age_group": clean_metadata.get("연령대", ""),
                "marriage": clean_metadata.get("결혼여부", ""),
                "children": clean_metadata.get("자녀수"),
                "family": clean_metadata.get("가족수", ""),
                "education": clean_metadata.get("최종학력", ""),
            },
            "welcome2_info": {
                "job": clean_metadata.get("직업", "") or original_job,
                "job_role": clean_metadata.get("직무", "") or original_job_role,
                "personal_income": clean_metadata.get("월평균 개인소득", "") or clean_metadata.get("개인소득", ""),
                "household_income": clean_metadata.get("월평균 가구소득", "") or clean_metadata.get("가구소득", ""),
            },
            "similarity": similarity,
            "embedding": None,
            "responses": responses,
            "aiSummary": ai_summary,
            "created_at": datetime.now().isoformat(),
            "metadata": clean_metadata
        })
    
    # ⭐ 페이지네이션은 이미 mb_sn_list 단계에서 적용됨
    # results는 이미 페이지네이션된 결과이므로 추가 페이지네이션 불필요
    import math
    pages = math.ceil(total_count / limit) if limit > 0 else 1
    
    convert_time = time.time() - convert_start_time
    logger.info(f"[Panel Details] 완료: {convert_time:.2f}초, 전체 {total_count}개 패널 중 현재 페이지 {len(results)}개 반환")
    
    return {
        "count": len(results),
        "total": total_count,  # 전체 검색 결과 개수
        "page": page,
        "page_size": limit,
        "pages": pages,
        "results": results  # 이미 페이지네이션된 결과
    }

