"""패널 상세 API 엔드포인트"""
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
import logging

from app.utils.merged_data_loader import get_panel_from_merged_db
from app.services.lifestyle_classifier import generate_lifestyle_summary
from app.core.config import ANTHROPIC_API_KEY

router = APIRouter()
logger = logging.getLogger(__name__)


class BatchPanelRequest(BaseModel):
    panel_ids: List[str]


@router.get("/api/panels/{panel_id}")
async def get_panel(
    panel_id: str
):
    """패널 상세 정보 조회 (NeonDB merged 테이블 + Pinecone 병합)"""
    logger.info(f"[Panel API] ========== 패널 상세 조회 시작: {panel_id} ==========")
    """
    패널 상세 정보 조회 (NeonDB merged.panel_data 테이블에서 기본 데이터 로드, Pinecone에서 응답 데이터 병합)
    
    Args:
        panel_id: 패널 ID (mb_sn)
        
    Returns:
        패널 상세 정보 (merged 데이터 + Pinecone 응답 데이터)
    """
    try:
        # 1. ⭐ NeonDB merged.panel_data 테이블에서 기본 데이터 조회
        merged_data = None
        try:
            merged_data = await get_panel_from_merged_db(panel_id)
            logger.info(f"[Panel API] merged 테이블 조회 결과: {panel_id}, 데이터 존재: {merged_data is not None}")
        except Exception as e:
            logger.warning(f"[Panel API] merged 테이블 조회 중 오류 발생: {panel_id}, 오류: {str(e)}", exc_info=True)
            # 에러가 발생해도 Pinecone에서 조회 시도
        
        if not merged_data:
            logger.error(f"[Panel API] merged 테이블에서 패널을 찾을 수 없음: {panel_id}")
            raise HTTPException(status_code=404, detail=f"Panel not found: {panel_id}")
        
        logger.info(f"[Panel API] merged 테이블에서 패널 데이터 조회 완료: {panel_id}")
        
        # 2. merged_data로부터 welcome1_info, welcome2_info 생성 및 기본 필드 매핑
        welcome1_info = {}
        welcome2_info = {}
        
        # 기본 필드 추출 (merged_data에서 직접)
        gender = merged_data.get("gender", "")
        age = merged_data.get("age", 0)
        location = merged_data.get("location", "")
        detail_location = merged_data.get("detail_location", "")
        region = location
        if detail_location:
            region = f"{location} {detail_location}".strip() if location else detail_location
        
        # 커버리지 계산 (merged_data에는 index 정보가 없으므로 Pinecone에서 가져옴)
        coverage = None
        try:
            from app.api.pinecone_panel_detail import get_panel_from_pinecone
            pinecone_data = get_panel_from_pinecone(panel_id)
            if pinecone_data and pinecone_data.get('coverage'):
                coverage = pinecone_data.get('coverage')
                logger.info(f"[Panel API] Pinecone에서 coverage 가져옴: {panel_id}, coverage={coverage}")
        except Exception as e:
            logger.warning(f"[Panel API] Pinecone에서 coverage 가져오기 실패: {panel_id}, 오류: {str(e)}")
            # coverage가 없어도 계속 진행
        
        # Welcome 1차 정보 (인구통계 정보)
        if merged_data:
            welcome1_info = {
                "gender": gender,
                "age": age if age else None,
                "region": location,
                "detail_location": detail_location,
                "age_group": merged_data.get("연령대", ""),
                "marriage": merged_data.get("결혼여부", ""),
                "children": merged_data.get("자녀수"),
                "family": merged_data.get("가족수", ""),
                "education": merged_data.get("최종학력", ""),
            }
            # 빈 값 제거 (하지만 0은 유지 - 자녀수 0명일 수 있음)
            welcome1_info = {k: v for k, v in welcome1_info.items() if v not in [None, ""]}
        
        # Welcome 2차 정보 (직업/소득 정보)
        if merged_data:
            welcome2_info = {
                "job": merged_data.get("직업", ""),
                "job_role": merged_data.get("직무", ""),
                "personal_income": merged_data.get("월평균 개인소득", ""),
                "household_income": merged_data.get("월평균 가구소득", ""),
            }
            # 빈 값 제거
            welcome2_info = {k: v for k, v in welcome2_info.items() if v not in [None, ""]}
        
        # 3. 데이터 구성: merged_data를 기본으로 사용
        result = {
            **merged_data,  # merged 데이터가 기본
        }
        
        # ⭐ 기본 필드 명시적 설정 (프론트엔드 호환성)
        result['gender'] = gender
        result['age'] = age if age else 0
        result['region'] = region
        result['name'] = merged_data.get('mb_sn', panel_id)
        
        # welcome1_info, welcome2_info 추가
        if welcome1_info:
            result['welcome1_info'] = welcome1_info
        if welcome2_info:
            result['welcome2_info'] = welcome2_info
        
        # ⭐ Pinecone 데이터는 더 이상 사용하지 않음 (응답이력은 merged의 quick_answers 사용)
        # 커버리지는 Pinecone에서 가져옴 (없으면 None)
        result['coverage'] = coverage
        
        # 5. 기본 필드 설정 (merged_data에서 가져온 데이터 기반)
        if 'mb_sn' in merged_data:
            result['id'] = merged_data['mb_sn']
        
        # 6. income 필드 설정
        if 'income' not in result or not result.get('income'):
            result['income'] = merged_data.get('월평균 개인소득', '') or merged_data.get('월평균 가구소득', '')
        
        # 7. created_at 필드 설정 (없으면 현재 시간)
        if 'created_at' not in result:
            from datetime import datetime
            result['created_at'] = datetime.now().isoformat()
        
        # 8. tags 필드 설정 (없으면 빈 배열)
        if 'tags' not in result:
            result['tags'] = []
        
        # 9. responses 필드 설정: quick_answers에서 qpoll 질문-답 변환 (카테고리별 그룹화)
        if 'responses' not in result or not result.get('responses'):
            # merged_data의 quick_answers를 responses 형식으로 변환
            quick_answers = merged_data.get('quick_answers', {})
            responses = []
            
            # 카테고리 매핑 (pinecone_topic → 카테고리 이름)
            # category_config.json의 pinecone_topic을 기준으로 매핑
            try:
                from app.core.config import load_category_config
                category_config = load_category_config()
                category_mapping = {}
                for cat_name, cat_info in category_config.items():
                    pinecone_topic = cat_info.get('pinecone_topic', '')
                    if pinecone_topic:
                        # 특별 처리: 자동차는 "모바일·자동차"로 매핑
                        if cat_name == '자동차':
                            category_mapping[pinecone_topic] = '모바일·자동차'
                        # 전자제품의 휴대폰 관련 내용도 "모바일·자동차"로 매핑 (별도 topic이 있는 경우)
                        elif cat_name == '전자제품' and '휴대폰' in cat_info.get('description', ''):
                            # 전자제품은 그대로 유지하되, 휴대폰 관련 topic이 별도로 있으면 모바일·자동차로 매핑
                            category_mapping[pinecone_topic] = cat_name
                        else:
                            category_mapping[pinecone_topic] = cat_name
                
                # 추가 매핑: 모바일 관련 topic이 별도로 있는 경우
                category_mapping['모바일'] = '모바일·자동차'
                category_mapping['휴대폰'] = '모바일·자동차'
                
            except Exception as e:
                logger.warning(f"[Panel API] 카테고리 설정 로드 실패, 기본 매핑 사용: {e}")
                category_mapping = {
                    '인구': '인구',
                    '직업소득': '직업소득',
                    '전자제품': '전자제품',
                    '자동차': '모바일·자동차',
                    '모바일': '모바일·자동차',
                    '휴대폰': '모바일·자동차',
                    '음주': '음주',
                    '흡연': '흡연',
                }
            
            # 날짜 정보 (created_at 사용, 없으면 현재 날짜)
            from datetime import datetime
            response_date = ''
            if result.get('created_at'):
                try:
                    if isinstance(result['created_at'], str):
                        dt = datetime.fromisoformat(result['created_at'].replace('Z', '+00:00'))
                    else:
                        dt = result['created_at']
                    response_date = dt.strftime('%Y.%m.%d')
                except:
                    response_date = datetime.now().strftime('%Y.%m.%d')
            else:
                response_date = datetime.now().strftime('%Y.%m.%d')
            
            if isinstance(quick_answers, dict) and quick_answers:
                logger.info(f"[Panel API] quick_answers 파싱 시작: {len(quick_answers)}개 키")
                
                # 질문 내용 기반 카테고리 매핑 함수
                def determine_category_from_question(question: str, answer: str = "") -> str:
                    """질문 내용을 분석하여 카테고리 결정"""
                    if not question:
                        return '기타'
                    
                    question_lower = question.lower()
                    answer_lower = answer.lower() if answer else ""
                    combined_text = f"{question_lower} {answer_lower}"
                    
                    # 인구 관련 키워드
                    if any(kw in combined_text for kw in ['지역', '나이', '연령', '성별', '결혼', '자녀', '가족', '학력', '최종학력']):
                        return '인구'
                    
                    # 직업소득 관련 키워드
                    if any(kw in combined_text for kw in ['직업', '직무', '소득', '월평균', '개인소득', '가구소득', '월급', '연봉']):
                        return '직업소득'
                    
                    # 전자제품 관련 키워드
                    if any(kw in combined_text for kw in ['전자제품', '가전', '냉장고', '세탁기', '에어컨', '청소기', 'TV', '텔레비전']):
                        return '전자제품'
                    
                    # 모바일·자동차 관련 키워드
                    if any(kw in combined_text for kw in ['휴대폰', '스마트폰', '핸드폰', '갤럭시', '아이폰', '자동차', '차량', '차', 'BMW', '현대', '기아']):
                        return '모바일·자동차'
                    
                    # 음주 관련 키워드
                    if any(kw in combined_text for kw in ['음주', '술', '소주', '맥주', '양주', '와인', '음용']):
                        return '음주'
                    
                    # 흡연 관련 키워드
                    if any(kw in combined_text for kw in ['흡연', '담배', '전자담배', '궐련', '가열식']):
                        return '흡연'
                    
                    return '기타'
                
                # ⭐ 실제 형식: {Q001: {question: "...", answer: "..."}, Q002: {...}}
                for q_key, q_data in quick_answers.items():
                    if isinstance(q_data, dict):
                        # {question: "...", answer: "..."} 형식
                        question = q_data.get('question', '') or q_data.get('질문', '')
                        answer = q_data.get('answer', '') or q_data.get('답변', '')
                        
                        if answer and str(answer).strip():
                            # 질문 내용 기반으로 카테고리 결정
                            category_name = determine_category_from_question(question, answer)
                            
                            responses.append({
                                'key': q_key,
                                'category': category_name,
                                'title': question if question else q_key,  # 질문을 제목으로 사용
                                'answer': str(answer).strip(),
                                'date': response_date
                            })
                    elif isinstance(q_data, str) and q_data.strip():
                        # 문자열 형식 (fallback)
                        category_name = determine_category_from_question("", q_data)
                        responses.append({
                            'key': q_key,
                            'category': category_name,
                            'title': q_key,
                            'answer': q_data.strip(),
                            'date': response_date
                        })
                    elif isinstance(q_data, list):
                        # 리스트 형식 (fallback)
                        for qa_item in q_data:
                            if isinstance(qa_item, dict):
                                question = qa_item.get('question', '') or qa_item.get('질문', '')
                                answer = qa_item.get('answer', '') or qa_item.get('답변', '')
                                if answer and str(answer).strip():
                                    category_name = determine_category_from_question(question, answer)
                                    responses.append({
                                        'key': q_key,
                                        'category': category_name,
                                        'title': question if question else q_key,
                                        'answer': str(answer).strip(),
                                        'date': response_date
                                    })
            
            # quick_answers가 비어있거나 responses가 없으면 "qpoll 응답 없음" 메시지 추가하지 않음
            # (프론트엔드에서 빈 상태로 표시)
            
            # Pinecone의 responses가 있으면 병합 (중복 제거)
            if 'responses' in result and isinstance(result['responses'], list):
                existing_keys = {r.get('key', '') for r in responses}
                for pinecone_resp in result['responses']:
                    resp_key = pinecone_resp.get('key', '')
                    if resp_key and resp_key not in existing_keys:
                        # Pinecone 응답에도 카테고리와 날짜 추가
                        pinecone_resp['category'] = pinecone_resp.get('category', resp_key)
                        pinecone_resp['date'] = pinecone_resp.get('date', response_date)
                        responses.append(pinecone_resp)
                        existing_keys.add(resp_key)
            
            result['responses'] = responses if responses else []
        
        # 10. metadata 필드 설정 (merged_data의 모든 필드를 metadata로도 포함)
        if merged_data:
            # 시스템 필드 제외하고 metadata 생성
            metadata = {}
            exclude_fields = ['mb_sn', 'quick_answers', 'id', 'name', 'gender', 'age', 'region', 'income']
            for key, value in merged_data.items():
                if key not in exclude_fields and value is not None:
                    metadata[key] = value
            if metadata:
                result['metadata'] = metadata
        
        logger.info(f"[Panel API] ========== 패널 상세 조회 완료: {panel_id} ==========")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Panel API] 패널 조회 실패: {panel_id}, 오류: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Panel fetch failed: {str(e)}")


@router.post("/api/panels/batch")
async def get_panels_batch(
    request: BatchPanelRequest
):
    """여러 패널의 상세 정보를 한 번에 조회 (배치 API)"""
    logger.info(f"[Panel API] ========== 배치 패널 조회 시작: {len(request.panel_ids)}개 ==========")
    
    try:
        if not request.panel_ids:
            return {"results": []}
        
        # NeonDB에서 배치로 조회
        from app.utils.merged_data_loader import get_panels_from_merged_db_batch
        
        # 배치로 merged 데이터 조회
        merged_data_map = await get_panels_from_merged_db_batch(request.panel_ids)
        
        results = []
        for panel_id in request.panel_ids:
            merged_data = merged_data_map.get(panel_id)
            
            if not merged_data:
                # merged 데이터가 없으면 기본 정보만 반환
                results.append({
                    "id": panel_id,
                    "name": panel_id,
                    "gender": "",
                    "age": 0,
                    "region": "",
                    "income": "",
                    "metadata": {},
                    "responses": []
                })
                continue
            
            # merged_data로부터 기본 필드 추출
            gender = merged_data.get("gender", "")
            age = merged_data.get("age", 0)
            location = merged_data.get("location", "")
            detail_location = merged_data.get("detail_location", "")
            region = location
            if detail_location:
                region = f"{location} {detail_location}".strip() if location else detail_location
            
            # metadata 필드 생성
            metadata = {}
            exclude_fields = ['mb_sn', 'quick_answers', 'id', 'name', 'gender', 'age', 'region', 'income']
            for key, value in merged_data.items():
                if key not in exclude_fields and value is not None:
                    metadata[key] = value
            
            # 결과 생성
            result = {
                "id": merged_data.get('mb_sn', panel_id),
                "name": merged_data.get('mb_sn', panel_id),
                "gender": gender,
                "age": age if age else 0,
                "region": region,
                "income": merged_data.get('월평균 개인소득', '') or merged_data.get('월평균 가구소득', ''),
                "metadata": metadata,
                "responses": []  # 배치 조회에서는 응답 제외 (필요시 개별 조회)
            }
            
            results.append(result)
        
        logger.info(f"[Panel API] ========== 배치 패널 조회 완료: {len(results)}개 ==========")
        return {"results": results}
        
    except Exception as e:
        logger.error(f"[Panel API] 배치 패널 조회 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch panel fetch failed: {str(e)}")


@router.get("/api/panels/{panel_id}/ai-summary")
async def get_panel_ai_summary(
    panel_id: str
):
    """패널 AI 요약 생성 (상세정보 열 때만 호출)"""
    logger.info(f"[Panel API] ========== AI 요약 생성 시작: {panel_id} ==========")
    """
    패널 AI 요약 생성 (라이프스타일 분류 기반)
    
    Args:
        panel_id: 패널 ID (mb_sn)
        
    Returns:
        AI 요약 텍스트
    """
    try:
        if not ANTHROPIC_API_KEY:
            raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        
        # 라이프스타일 분류 기반 요약 생성
        summary = generate_lifestyle_summary(panel_id, ANTHROPIC_API_KEY)
        
        logger.info(f"[Panel API] ========== AI 요약 생성 완료: {panel_id} ==========")
        return {
            "panel_id": panel_id,
            "aiSummary": summary
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Panel API] AI 요약 생성 실패: {panel_id}, 오류: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI summary generation failed: {str(e)}")
