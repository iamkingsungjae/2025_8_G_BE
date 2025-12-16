"""Pinecone에서 단일 패널 상세 정보 가져오기 (Pinecone 메타데이터만 사용)"""
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import numpy as np

from app.core.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, load_category_config
from app.utils.merged_data_loader import load_merged_data
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


def get_panel_from_pinecone(panel_id: str) -> Optional[Dict[str, Any]]:
    """
    Pinecone에서 패널 상세 정보 조회 (Pinecone 메타데이터만 사용)
    
    Pinecone 구조:
    - 각 벡터는 topic, index, mb_sn, text와 카테고리별 메타데이터를 포함
    - topic: 카테고리별 구분 (인구, 직업소득, 모바일·자동차 등)
    - index: 데이터 소스 구분 (welcome_1st/1, welcome_2nd/2, quick_answer/3)
    - mb_sn: 패널 ID
    - text: 자연어 텍스트 (임베딩 원본)
    
    Args:
        panel_id: 패널 ID (mb_sn)
        
    Returns:
        패널 상세 정보 딕셔너리 또는 None
    """
    try:
        logger.info(f"[Panel Detail] 패널 상세 정보 조회 시작: {panel_id}")
        category_config = load_category_config()
        
        # Pinecone 인덱스 연결
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # 인덱스 차원 동적으로 가져오기
        stats = index.describe_index_stats()
        dimension = stats.get('dimension', 1536)
        
        # 랜덤 벡터 생성 (메타데이터만 필요하므로)
        random_vector = np.random.rand(dimension).astype(np.float32).tolist()
        norm = np.linalg.norm(random_vector)
        if norm > 0:
            random_vector = (np.array(random_vector) / norm).tolist()
        
        # ⭐ 효율적인 조회: mb_sn으로 모든 topic 한 번에 조회
        # Pinecone은 필터로 mb_sn만 지정하면 모든 topic의 데이터를 가져올 수 있음
        all_results = index.query(
            vector=random_vector,
            top_k=1000,  # 충분히 많이 가져오기 (한 패널의 모든 topic 데이터)
            include_metadata=True,
            filter={
                "mb_sn": panel_id
            }
        )
        
        if not all_results.matches:
            logger.warning(f"[Panel Detail] 패널을 찾을 수 없음: {panel_id}")
            return None
        
        logger.info(f"[Panel Detail] {len(all_results.matches)}개 벡터 발견")
        
        # 디버그: 첫 번째 매치의 메타데이터 샘플 확인
        if all_results.matches:
            first_match = all_results.matches[0]
            first_metadata = first_match.metadata or {}
            logger.info(f"[Panel Detail] 첫 번째 매치 샘플 - topic: {first_metadata.get('topic')}, index: {first_metadata.get('index')}, mb_sn: {first_metadata.get('mb_sn')}")
            logger.info(f"[Panel Detail] 첫 번째 매치 메타데이터 키: {list(first_metadata.keys())[:10]}")
        
        # 메타데이터 수집 및 구조화 (topic 기반만 사용)
        metadata_by_topic = {}  # topic별 메타데이터 그룹화
        all_metadata = {}  # 모든 메타데이터 병합
        index_values = []  # 커버리지 계산용 (welcome1, welcome2, quick_answer)
        
        # 모든 매치를 topic별로 그룹화
        for match in all_results.matches:
            match_metadata = match.metadata or {}
            if not match_metadata:
                continue
                
            topic = match_metadata.get("topic", "unknown")
            
            # index 값 수집 (커버리지 계산용)
            index_val = match_metadata.get("index")
            if index_val:
                # 문자열 또는 숫자 모두 처리
                if isinstance(index_val, (int, str)):
                    index_str = str(index_val)
                    # 정규화: welcome_1st/1 -> w1, welcome_2nd/2 -> w2, quick_answer/3 -> q
                    if index_str in ["1", "welcome_1st"] and "w1" not in index_values:
                        index_values.append("w1")
                    elif index_str in ["2", "welcome_2nd"] and "w2" not in index_values:
                        index_values.append("w2")
                    elif index_str in ["3", "quick_answer"] and "q" not in index_values:
                        index_values.append("q")
            
            # topic별로 메타데이터 그룹화
            if topic not in metadata_by_topic:
                metadata_by_topic[topic] = []
            metadata_by_topic[topic].append(match_metadata)
            
            # ⭐ 모든 메타데이터 병합 (시스템 필드 제외)
            # 각 topic의 모든 메타데이터를 수집하여 병합
            for key, value in match_metadata.items():
                # 시스템 필드 제외
                if key in ["topic", "index", "mb_sn"]:
                    continue
                
                # None만 제외 (빈 문자열은 나중에 clean_metadata에서 필터링)
                if value is None:
                    continue
                
                # 메타데이터 병합 전략
                if key not in all_metadata:
                    # 새로운 키면 그대로 추가 (빈 문자열도 일단 추가)
                    all_metadata[key] = value
                else:
                    # 기존 키가 있는 경우
                    existing_value = all_metadata[key]
                    
                    # 리스트인 경우 병합
                    if isinstance(existing_value, list):
                        if isinstance(value, list):
                            # 두 리스트 병합 (중복 제거)
                            merged = list(set(existing_value + value))
                            all_metadata[key] = merged
                        elif value not in existing_value:
                            existing_value.append(value)
                    elif isinstance(value, list):
                        # 기존 값이 리스트가 아니고 새 값이 리스트인 경우
                        all_metadata[key] = [existing_value] + value
                    else:
                        # 둘 다 단일 값인 경우, 빈 문자열이 아닌 값 우선
                        if isinstance(value, str) and not value.strip():
                            # 새 값이 빈 문자열이면 기존 값 유지
                            pass
                        elif isinstance(existing_value, str) and not existing_value.strip():
                            # 기존 값이 빈 문자열이면 새 값으로 교체
                            all_metadata[key] = value
                        else:
                            # 둘 다 유효한 값이면 나중 값으로 덮어쓰기 (더 최신 정보)
                            all_metadata[key] = value
        
        logger.info(f"[Panel Detail] 발견된 topic: {list(metadata_by_topic.keys())}")
        logger.info(f"[Panel Detail] topic별 메타데이터 개수: {[(t, len(m)) for t, m in metadata_by_topic.items()]}")
        
        # 기본 정보 추출 (인구 topic 우선, 없으면 전체 메타데이터에서)
        인구_topic = category_config.get("기본정보", {}).get("pinecone_topic", "인구")
        인구_metadata = {}
        
        if 인구_topic in metadata_by_topic:
            # 인구 topic의 첫 번째 메타데이터 사용 (일반적으로 하나만 있음)
            인구_metadata = metadata_by_topic[인구_topic][0] if metadata_by_topic[인구_topic] else {}
        
        # 기본 정보는 인구 topic 우선, 없으면 전체 메타데이터에서
        gender = 인구_metadata.get("성별") or all_metadata.get("성별", "")
        region = 인구_metadata.get("지역") or all_metadata.get("지역", "")
        detail_location = 인구_metadata.get("지역구") or all_metadata.get("지역구", "")
        
        if detail_location and region:
            region = f"{region} {detail_location}"
        elif detail_location:
            region = detail_location
        
        age = 0
        age_value = 인구_metadata.get("나이") or all_metadata.get("나이")
        if age_value:
            try:
                age = int(float(age_value))
            except (ValueError, TypeError):
                pass
        
        # 연령대에서 나이 추정 (나이가 없을 때)
        if not age:
            age_group = 인구_metadata.get("연령대") or all_metadata.get("연령대", "")
            if age_group and "대" in age_group:
                try:
                    age_base = int(age_group.replace("대", ""))
                    age = age_base + 5  # 중간값 사용
                except (ValueError, TypeError):
                    pass
        
        # ⭐ 학력 필드명 통일: Pinecone에는 "학력"으로 저장되지만 프론트엔드는 "최종학력"을 기대
        # all_metadata에 "최종학력" 키로도 추가
        if "학력" in 인구_metadata and "최종학력" not in all_metadata:
            all_metadata["최종학력"] = 인구_metadata["학력"]
        elif "학력" in all_metadata and "최종학력" not in all_metadata:
            all_metadata["최종학력"] = all_metadata["학력"]
        
        # 소득 정보 (직업소득 topic에서)
        # ⭐ Pinecone에는 개인소득_min/max, 가구소득_min/max로 저장됨
        income = ""
        직업소득_topic = category_config.get("직업소득", {}).get("pinecone_topic", "직업소득")
        
        if 직업소득_topic in metadata_by_topic:
            직업소득_metadata = metadata_by_topic[직업소득_topic][0] if metadata_by_topic[직업소득_topic] else {}
            # 개인소득_min/max를 개인소득으로 변환
            if "개인소득_min" in 직업소득_metadata and "개인소득_max" in 직업소득_metadata:
                income = f"{int(직업소득_metadata['개인소득_min'])}~{int(직업소득_metadata['개인소득_max'])}만원"
            elif "개인소득_min" in 직업소득_metadata:
                income = f"{int(직업소득_metadata['개인소득_min'])}만원 이상"
            elif "개인소득_max" in 직업소득_metadata:
                income = f"{int(직업소득_metadata['개인소득_max'])}만원 이하"
            # 가구소득_min/max를 가구소득으로 변환
            elif "가구소득_min" in 직업소득_metadata and "가구소득_max" in 직업소득_metadata:
                income = f"{int(직업소득_metadata['가구소득_min'])}~{int(직업소득_metadata['가구소득_max'])}만원"
            elif "가구소득_min" in 직업소득_metadata:
                income = f"{int(직업소득_metadata['가구소득_min'])}만원 이상"
            elif "가구소득_max" in 직업소득_metadata:
                income = f"{int(직업소득_metadata['가구소득_max'])}만원 이하"
            # 기존 필드명도 확인 (하위 호환성)
            elif 직업소득_metadata.get("개인소득"):
                income = str(직업소득_metadata["개인소득"])
            elif 직업소득_metadata.get("가구소득"):
                income = str(직업소득_metadata["가구소득"])
        else:
            # 전체 메타데이터에서 확인
            if "개인소득_min" in all_metadata and "개인소득_max" in all_metadata:
                income = f"{int(all_metadata['개인소득_min'])}~{int(all_metadata['개인소득_max'])}만원"
            elif all_metadata.get("개인소득"):
                income = str(all_metadata["개인소득"])
            elif all_metadata.get("가구소득"):
                income = str(all_metadata["가구소득"])
        
        # ⭐ 응답 수집: topic별로 구조화 (index 구분 포함)
        # topic별로 응답을 그룹화하여 프론트엔드에서 topic 기반 UI 구성 가능하도록
        responses_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        all_responses = []  # 하위 호환성을 위한 전체 응답 리스트
        
        for match in all_results.matches:
            match_metadata = match.metadata or {}
            text = match_metadata.get("text", "")
            topic = match_metadata.get("topic", "기타")
            index_val = match_metadata.get("index", "")
            
            # text가 있고 무응답이 아닌 경우만 수집
            if text and not _is_no_response(text):
                # index가 quick_answer(3)인 경우 질문 추출 시도
                is_quick_answer = str(index_val) in ["3", "quick_answer"]
                
                # 질문 추출 (text에서 질문 부분이 있는 경우)
                question = None
                answer = text
                
                # text에서 질문과 답변을 구분 (다양한 형식 지원)
                if is_quick_answer:
                    # 형식 1: "질문: ... 답변: ..." 또는 "질문 ... 답변 ..."
                    if "질문" in text and "답변" in text:
                        # "답변"으로 분리
                        parts = text.split("답변", 1)  # 최대 1번만 분리
                        if len(parts) > 1:
                            question_part = parts[0].replace("질문", "").strip()
                            # 콜론 제거
                            question = question_part.replace(":", "").strip()
                            answer_part = parts[1].strip()
                            # 콜론 제거
                            answer = answer_part.replace(":", "").strip()
                    # 형식 2: "Q: ... A: ..." 또는 "Q ... A ..."
                    elif ("Q:" in text or text.strip().startswith("Q ")) and ("A:" in text or "답변" in text):
                        if "A:" in text:
                            parts = text.split("A:", 1)
                        elif "답변" in text:
                            parts = text.split("답변", 1)
                        else:
                            parts = None
                        if parts and len(parts) > 1:
                            question_part = parts[0].replace("Q:", "").replace("Q", "").strip()
                            question = question_part.strip()
                            answer = parts[1].strip()
                    # 형식 3: "..."? "..." (물음표로 구분) - 질문이 물음표로 끝나는 경우
                    elif "?" in text:
                        q_idx = text.index("?")
                        # 물음표 앞부분이 질문일 가능성이 높음
                        potential_question = text[:q_idx + 1].strip()
                        potential_answer = text[q_idx + 1:].strip()
                        
                        # 질문이 너무 짧으면 (5자 이하) 물음표가 답변 안에 있을 수 있음
                        if len(potential_question) > 5:
                            question = potential_question
                            answer = potential_answer.replace(":", "").strip()
                            if not answer or len(answer) < 3:
                                # 답변이 너무 짧으면 전체를 답변으로
                                answer = text
                                question = None
                        else:
                            # 질문이 너무 짧으면 질문이 없는 것으로 간주
                            question = None
                            answer = text
                    # 형식 4: 메타데이터에서 질문 정보 확인 (NeonDB quick_answers에서 가져올 수 있음)
                    # 현재는 text만 사용하므로 나중에 개선 가능
                    else:
                        # 질문 형식을 찾지 못한 경우, 전체를 답변으로
                        question = None
                        answer = text
                
                response_item = {
                    "key": f"response_{len(all_responses)}",
                    "title": topic,
                    "question": question if is_quick_answer and question else None,  # quick_answer인 경우만 질문 포함
                    "answer": answer,
                    "index": index_val,  # 커버리지 표시용
                    "is_quick_answer": is_quick_answer
                }
                
                # topic별로 그룹화
                if topic not in responses_by_topic:
                    responses_by_topic[topic] = []
                responses_by_topic[topic].append(response_item)
                
                # 전체 응답 리스트에도 추가 (하위 호환성)
                all_responses.append(response_item)
        
        # ⭐ quick_answer 응답이 없는 경우 메시지 추가
        has_quick_answer = any(
            response.get('is_quick_answer', False) 
            for responses in responses_by_topic.values() 
            for response in responses
        )
        
        if not has_quick_answer:
            # qpoll 응답이 없음을 알리는 메시지 추가
            no_qpoll_response = {
                "key": "no_qpoll",
                "title": "Quick Poll",
                "question": "Quick Poll",
                "answer": "이 패널은 qpoll에 대해 응답하지 않았습니다.",
                "index": 3,
                "is_quick_answer": True
            }
            # responses_by_topic에 추가 (기본 topic으로)
            if "기타" not in responses_by_topic:
                responses_by_topic["기타"] = []
            responses_by_topic["기타"].append(no_qpoll_response)
            all_responses.append(no_qpoll_response)
        
        # 응답이 없을 때만 메시지 표시
        if not all_responses:
            all_responses = [{
                "key": "no_response",
                "title": "응답 없음",
                "answer": "해당 패널의 응답 정보가 없습니다."
            }]
        
        # ⭐ 커버리지 계산
        coverage = None
        has_w1 = "w1" in index_values
        has_w2 = "w2" in index_values
        has_q = "q" in index_values
        
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
        
        logger.info(f"[Panel Detail] 커버리지 계산: index_values={index_values}, coverage={coverage}")
        
        # ⭐ topic별 메타데이터도 구조화 (프론트엔드에서 topic별 섹션 구성용)
        # metadata_by_topic은 이미 수집되어 있음
        
        # 태그는 Pinecone에 없으므로 빈 배열
        tags = []
        
        # 근거 추출 (주요 topic의 text 필드)
        evidence = []
        # 인구 topic의 text 우선, 없으면 첫 번째 text 사용
        if 인구_topic in metadata_by_topic and metadata_by_topic[인구_topic]:
            text = metadata_by_topic[인구_topic][0].get("text", "")
            if text and not _is_no_response(text):
                evidence.append({
                    "text": text[:500],
                    "source": "pinecone",
                    "similarity": None
                })
        else:
            # 다른 topic의 text 사용
            for match in all_results.matches:
                match_metadata = match.metadata or {}
                text = match_metadata.get("text", "")
                if text and not _is_no_response(text):
                    evidence.append({
                        "text": text[:500],
                        "source": "pinecone",
                        "similarity": None
                    })
                    break  # 첫 번째 유효한 text만 사용
        
        # AI 요약 (evidence의 text 사용)
        ai_summary = ""
        if evidence and evidence[0].get("text"):
            ai_summary = evidence[0]["text"][:300] + "..." if len(evidence[0]["text"]) > 300 else evidence[0]["text"]
        
        if not ai_summary:
            ai_summary = "요약 정보가 없습니다."
        
        # welcome1, welcome2 정보 추출 (개요에 표시용)
        welcome1_info = {}
        welcome2_info = {}
        
        # ⭐ merged_data 로드 및 original_panel 초기화 (함수 시작 부분에서)
        merged_data = load_merged_data()
        original_panel = {}
        if merged_data and panel_id in merged_data:
            original_panel = merged_data[panel_id]
        
        # welcome1 정보 (인구 topic에서)
        # ⭐ has_welcome1이 False여도 인구 topic이 있으면 welcome1_info 생성
        if 인구_topic in metadata_by_topic:
            인구_data = metadata_by_topic[인구_topic][0] if metadata_by_topic[인구_topic] else {}
            logger.info(f"[Panel Detail] 인구 topic 메타데이터 키: {list(인구_data.keys())}")
            welcome1_info = {
                "gender": 인구_data.get("성별") or all_metadata.get("성별") or gender,
                "age": 인구_data.get("나이") or all_metadata.get("나이") or age,
                "region": 인구_data.get("지역") or all_metadata.get("지역") or region,
                "detail_location": 인구_data.get("지역구") or all_metadata.get("지역구") or detail_location,
                "age_group": 인구_data.get("연령대") or all_metadata.get("연령대", ""),
                "marriage": 인구_data.get("결혼여부") or all_metadata.get("결혼여부", ""),
                "children": 인구_data.get("자녀수") or all_metadata.get("자녀수"),  # ⚠️ Pinecone에 없을 수 있음
                "family": 인구_data.get("가족수") or all_metadata.get("가족수", ""),
                # ⭐ 학력 필드명 통일: Pinecone에는 "학력"으로 저장되지만 프론트엔드는 "최종학력"을 기대
                "education": 인구_data.get("학력") or 인구_data.get("최종학력") or all_metadata.get("학력") or all_metadata.get("최종학력", ""),
            }
            logger.info(f"[Panel Detail] welcome1_info 생성: {welcome1_info}")
        
        # welcome2 정보 (직업소득 topic에서)
        # ⭐ has_welcome2가 False여도 직업소득 topic이 있으면 welcome2_info 생성
        if 직업소득_topic in metadata_by_topic:
            직업소득_data = metadata_by_topic[직업소득_topic][0] if metadata_by_topic[직업소득_topic] else {}
            logger.info(f"[Panel Detail] 직업소득 topic 메타데이터 키: {list(직업소득_data.keys())}")
            
            # ⭐ 소득 필드 변환: 개인소득_min/max → 개인소득, 가구소득_min/max → 가구소득
            personal_income_str = ""
            if "개인소득_min" in 직업소득_data and "개인소득_max" in 직업소득_data:
                personal_income_str = f"{int(직업소득_data['개인소득_min'])}~{int(직업소득_data['개인소득_max'])}만원"
            elif "개인소득_min" in 직업소득_data:
                personal_income_str = f"{int(직업소득_data['개인소득_min'])}만원 이상"
            elif "개인소득_max" in 직업소득_data:
                personal_income_str = f"{int(직업소득_data['개인소득_max'])}만원 이하"
            else:
                # 기존 필드명도 확인 (하위 호환성)
                personal_income_str = 직업소득_data.get("개인소득") or all_metadata.get("개인소득", "")
            
            household_income_str = ""
            if "가구소득_min" in 직업소득_data and "가구소득_max" in 직업소득_data:
                household_income_str = f"{int(직업소득_data['가구소득_min'])}~{int(직업소득_data['가구소득_max'])}만원"
            elif "가구소득_min" in 직업소득_data:
                household_income_str = f"{int(직업소득_data['가구소득_min'])}만원 이상"
            elif "가구소득_max" in 직업소득_data:
                household_income_str = f"{int(직업소득_data['가구소득_max'])}만원 이하"
            else:
                # 기존 필드명도 확인 (하위 호환성)
                household_income_str = 직업소득_data.get("가구소득") or all_metadata.get("가구소득", "")
            
            # ⭐ original_panel은 이미 위에서 로드됨
            original_job = original_panel.get("직업", "")
            original_job_role = original_panel.get("직무", "")
            
            welcome2_info = {
                "job": 직업소득_data.get("직업") or all_metadata.get("직업", "") or original_job,
                "job_role": 직업소득_data.get("직무") or all_metadata.get("직무", "") or original_job_role,
                "personal_income": personal_income_str,
                "household_income": household_income_str,
            }
            logger.info(f"[Panel Detail] welcome2_info 생성: {welcome2_info}")
        
        # Pinecone 메타데이터만 반환 (모든 topic의 메타데이터 포함)
        # ⭐ 인구 topic과 직업소득 topic의 메타데이터를 우선적으로 포함
        clean_metadata = {}
        
        # 1단계: 인구 topic 메타데이터 우선 추가
        if 인구_topic in metadata_by_topic:
            인구_data = metadata_by_topic[인구_topic][0] if metadata_by_topic[인구_topic] else {}
            logger.info(f"[Panel Detail] 인구 topic 메타데이터 키: {list(인구_data.keys())}")
            for key, value in 인구_data.items():
                if key not in ["topic", "index", "mb_sn", "text"]:
                    if value is not None:
                        if isinstance(value, str) and value.strip():
                            clean_metadata[key] = value
                        elif not isinstance(value, str):
                            clean_metadata[key] = value
            # ⭐ 학력 → 최종학력 매핑 (프론트엔드 호환성)
            if "학력" in clean_metadata and "최종학력" not in clean_metadata:
                clean_metadata["최종학력"] = clean_metadata["학력"]
            # ⭐ 결혼여부, 가족수 필드명 확인 및 추가 (all_metadata에서도 확인)
            if "결혼여부" not in clean_metadata and "결혼여부" in all_metadata:
                clean_metadata["결혼여부"] = all_metadata["결혼여부"]
            if "가족수" not in clean_metadata and "가족수" in all_metadata:
                clean_metadata["가족수"] = all_metadata["가족수"]
        
        # 2단계: 직업소득 topic 메타데이터 추가
        if 직업소득_topic in metadata_by_topic:
            직업소득_data = metadata_by_topic[직업소득_topic][0] if metadata_by_topic[직업소득_topic] else {}
            logger.info(f"[Panel Detail] 직업소득 topic 메타데이터 키: {list(직업소득_data.keys())}")
            for key, value in 직업소득_data.items():
                if key not in ["topic", "index", "mb_sn", "text"]:
                    if value is not None:
                        if isinstance(value, str) and value.strip():
                            clean_metadata[key] = value
                        elif not isinstance(value, str):
                            clean_metadata[key] = value
            
            # ⭐ 소득 필드 변환: 개인소득_min/max → 개인소득, 가구소득_min/max → 가구소득
            if "개인소득_min" in clean_metadata and "개인소득_max" in clean_metadata:
                clean_metadata["개인소득"] = f"{int(clean_metadata['개인소득_min'])}~{int(clean_metadata['개인소득_max'])}만원"
            elif "개인소득_min" in clean_metadata:
                clean_metadata["개인소득"] = f"{int(clean_metadata['개인소득_min'])}만원 이상"
            elif "개인소득_max" in clean_metadata:
                clean_metadata["개인소득"] = f"{int(clean_metadata['개인소득_max'])}만원 이하"
            
            if "가구소득_min" in clean_metadata and "가구소득_max" in clean_metadata:
                clean_metadata["가구소득"] = f"{int(clean_metadata['가구소득_min'])}~{int(clean_metadata['가구소득_max'])}만원"
            elif "가구소득_min" in clean_metadata:
                clean_metadata["가구소득"] = f"{int(clean_metadata['가구소득_min'])}만원 이상"
            elif "가구소득_max" in clean_metadata:
                clean_metadata["가구소득"] = f"{int(clean_metadata['가구소득_max'])}만원 이하"
            
            # ⭐ 월평균 개인소득, 월평균 가구소득 필드도 추가 (프론트엔드 호환성)
            if "개인소득" in clean_metadata:
                clean_metadata["월평균 개인소득"] = clean_metadata["개인소득"]
            if "가구소득" in clean_metadata:
                clean_metadata["월평균 가구소득"] = clean_metadata["가구소득"]
        
        # 3단계: 나머지 모든 topic의 메타데이터 추가 (인구, 직업소득 제외)
        # ⭐ 모든 topic의 메타데이터를 clean_metadata에 추가
        for key, value in all_metadata.items():
            # 시스템 필드 제외
            if key in ["topic", "index", "mb_sn", "quickpoll_status", "text"]:
                continue
            # 이미 clean_metadata에 있는 키는 건너뛰기 (인구, 직업소득 우선)
            if key in clean_metadata:
                continue
            # None 제외
            if value is None:
                continue
            # 빈 문자열 제외 (하지만 숫자 0, False 등은 허용)
            if isinstance(value, str) and not value.strip():
                continue
            # 빈 리스트 제외
            if isinstance(value, list) and len(value) == 0:
                continue
            # 빈 딕셔너리 제외
            if isinstance(value, dict) and len(value) == 0:
                continue
            # ⭐ 모든 유효한 메타데이터 추가
            clean_metadata[key] = value
        
        # 디버그: 수집된 메타데이터 로깅
        logger.info(f"[Panel Detail] ========== 메타데이터 수집 결과 ==========")
        logger.info(f"[Panel Detail] all_metadata 키 개수: {len(all_metadata)}")
        logger.info(f"[Panel Detail] all_metadata 전체 키: {list(all_metadata.keys())}")
        logger.info(f"[Panel Detail] clean_metadata 키 개수: {len(clean_metadata)}")
        logger.info(f"[Panel Detail] clean_metadata 전체 키: {list(clean_metadata.keys())}")
        
        # ⭐ merged_final.json의 모든 필드를 clean_metadata에 병합 (Pinecone 메타데이터보다 우선)
        # 시스템 필드 제외하고 모든 데이터 추가
        if original_panel:
            for key, value in original_panel.items():
                # mb_sn은 이미 있으므로 제외
                if key == "mb_sn":
                    continue
                # None, 빈 문자열, 빈 리스트 제외
                if value is None:
                    continue
                if isinstance(value, str) and not value.strip():
                    continue
                if isinstance(value, list) and len(value) == 0:
                    continue
                # merged_final.json의 데이터가 우선 (Pinecone 메타데이터보다)
                clean_metadata[key] = value
        
        # 인구통계 관련 메타데이터 확인
        인구통계_키 = ["결혼여부", "자녀수", "가족수", "최종학력", "직업", "직무", "연령대", "성별", "지역", "나이"]
        인구통계_존재_all = [key for key in 인구통계_키 if key in all_metadata]
        인구통계_존재_clean = [key for key in 인구통계_키 if key in clean_metadata]
        logger.info(f"[Panel Detail] 인구통계 메타데이터 (all_metadata): {인구통계_존재_all}")
        logger.info(f"[Panel Detail] 인구통계 메타데이터 (clean_metadata): {인구통계_존재_clean}")
        
        # 흡연, 전자제품, 소득 관련 키 확인
        추가정보_키 = ["흡연경험", "흡연경험 담배브랜드", "궐련형 전자담배/가열식 전자담배 이용경험", 
                      "음용경험 술", "보유전제품", "보유 휴대폰 단말기 브랜드", "보유 휴대폰 모델명",
                      "보유차량여부", "자동차 제조사", "자동차 모델", "개인소득", "가구소득", 
                      "개인소득_min", "개인소득_max", "가구소득_min", "가구소득_max"]
        추가정보_존재_all = [key for key in 추가정보_키 if key in all_metadata]
        추가정보_존재_clean = [key for key in 추가정보_키 if key in clean_metadata]
        logger.info(f"[Panel Detail] 추가 정보 메타데이터 (all_metadata): {추가정보_존재_all}")
        logger.info(f"[Panel Detail] 추가 정보 메타데이터 (clean_metadata): {추가정보_존재_clean}")
        
        # 샘플 메타데이터 값 확인
        if clean_metadata:
            sample_keys = list(clean_metadata.keys())[:10]
            for key in sample_keys:
                logger.info(f"[Panel Detail] 샘플 메타데이터 - {key}: {clean_metadata[key]}")
        
        result = {
            "id": panel_id,
            "name": panel_id,
            "gender": gender,
            "age": age,
            "region": region,
            "income": str(income) if income else "",
            "coverage": coverage,  # ⭐ 커버리지 추가
            "tags": tags,
            "responses": all_responses,  # 하위 호환성 유지
            "responses_by_topic": responses_by_topic,  # ⭐ topic별 구조화된 응답
            "metadata_by_topic": {  # ⭐ topic별 구조화된 메타데이터
                topic: data[0] if data else {} 
                for topic, data in metadata_by_topic.items()
            },
            "merged_data": original_panel,  # ⭐ merged_final.json의 모든 원본 데이터
            "evidence": evidence,
            "aiSummary": ai_summary,
            "created_at": datetime.now().isoformat(),
            "metadata": clean_metadata,
            "welcome1_info": welcome1_info,
            "welcome2_info": welcome2_info,
        }
        
        logger.info(f"[Panel Detail] 패널 상세 정보 조회 완료: {panel_id}")
        return result
        
    except Exception as e:
        logger.error(f"[Panel Detail] 패널 상세 정보 조회 실패: {panel_id}, 오류: {e}", exc_info=True)
        return None
