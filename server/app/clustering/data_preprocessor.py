"""
클러스터링용 데이터 전처리
DB에서 추출한 원시 데이터를 클러스터링에 사용 가능한 형태로 변환
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_feature_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    DataFrame의 피처 타입을 자동으로 분류
    
    Returns:
    --------
    dict
        {'bin_cols': [...], 'cat_cols': [...], 'num_cols': [...]}
    """
    bin_cols = []
    cat_cols = []
    num_cols = []
    
    for col in df.columns:
        # mb_sn, cluster 등 제외
        if col in ['mb_sn', 'cluster', 'w2_data', 'qa_answers', 'data_text', 'answers_text']:
            continue
        
        # 이진 변수 패턴
        if col.startswith(('has_', 'is_', 'gender_', 'Q1_', 'Q5_', 'Q8_', 'family_type_')):
            bin_cols.append(col)
        # 범주형 변수
        elif col in ['region_lvl1', 'age_group', 'generation', 'Q6_label', 'Q4_label', 'Q1_label', 'education_level']:
            cat_cols.append(col)
        # 연속형 변수
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            # 이진 변수가 아닌 숫자형
            if col.endswith('_scaled') or col in ['age', 'age_z', 'income_personal', 'income_household']:
                num_cols.append(col)
            elif 'Q8_' in col and col not in bin_cols:
                num_cols.append(col)
            elif col not in bin_cols:
                # 기타 숫자형
                unique_values = df[col].nunique()
                if unique_values == 2:  # 이진 변수일 가능성
                    bin_cols.append(col)
                elif unique_values < 10:  # 범주형일 가능성
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
    
    return {
        'bin_cols': bin_cols,
        'cat_cols': cat_cols,
        'num_cols': num_cols
    }


def preprocess_for_clustering(
    raw_data: List[Dict[str, Any]],
    verbose: bool = False
) -> pd.DataFrame:
    """
    원시 데이터를 클러스터링용 DataFrame으로 전처리
    
    Parameters:
    -----------
    raw_data : List[Dict[str, Any]]
        DB에서 추출한 원시 데이터
    verbose : bool
        상세 로그 출력 여부
    
    Returns:
    --------
    pd.DataFrame
        전처리된 DataFrame
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not raw_data:
        logger.warning("[전처리] 원시 데이터가 비어있습니다.")
        return pd.DataFrame()
    
    logger.info(f"[전처리 시작] 원시 데이터: {len(raw_data)}개")
    df = pd.DataFrame(raw_data)
    logger.info(f"[전처리] DataFrame 생성: {len(df)}행, {len(df.columns)}열")
    
    # mb_sn 컬럼명 통일 (panel_id -> mb_sn)
    if 'panel_id' in df.columns and 'mb_sn' not in df.columns:
        df['mb_sn'] = df['panel_id']
    
    if 'mb_sn' not in df.columns:
        raise ValueError("mb_sn 또는 panel_id 컬럼이 필요합니다.")
    
    # JSON 데이터 파싱
    if 'w2_data' in df.columns or 'data_text' in df.columns:
        df = _parse_json_data(df)
    
    # 기본 피처 생성
    df = _create_basic_features(df)
    
    # 스케일링된 피처 생성 (가능한 경우)
    df = _create_scaled_features(df)
    
    if verbose:
        print(f"전처리 완료: {len(df)}행, {len(df.columns)}열")
        print(f"사용 가능한 피처: {[c for c in df.columns if c not in ['mb_sn', 'w2_data', 'qa_answers', 'data_text', 'answers_text']]}")
    
    logger.info(f"[전처리 완료] 최종 데이터: {len(df)}행, {len(df.columns)}열")
    logger.info(f"[전처리 완료] 사용 가능한 피처: {len([c for c in df.columns if c not in ['mb_sn', 'w2_data', 'qa_answers', 'data_text', 'answers_text']])}개")
    
    return df


def _parse_json_data(df: pd.DataFrame) -> pd.DataFrame:
    """JSON 데이터 파싱"""
    # w2_data 파싱
    if 'w2_data' in df.columns:
        for idx, row in df.iterrows():
            if pd.notna(row.get('w2_data')):
                try:
                    if isinstance(row['w2_data'], str):
                        data = json.loads(row['w2_data'])
                    else:
                        data = row['w2_data']
                    
                    # 필요한 필드 추출
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key not in df.columns:
                                df.at[idx, key] = value
                except:
                    pass
    
    # qa_answers 파싱
    if 'qa_answers' in df.columns or 'answers_text' in df.columns:
        answers_col = 'qa_answers' if 'qa_answers' in df.columns else 'answers_text'
        for idx, row in df.iterrows():
            if pd.notna(row.get(answers_col)):
                try:
                    if isinstance(row[answers_col], str):
                        # JSON 문자열 파싱
                        if row[answers_col].strip().startswith('{'):
                            answers = json.loads(row[answers_col])
                        else:
                            # 빈 문자열이거나 JSON이 아닌 경우 스킵
                            continue
                    else:
                        answers = row[answers_col]
                    
                    if isinstance(answers, dict):
                        for key, value in answers.items():
                            # Q001, Q002 형태를 Q1, Q2로 변환
                            if isinstance(key, str) and key.startswith('Q') and len(key) > 1:
                                # Q001 -> Q1, Q006 -> Q6 등
                                try:
                                    num = int(key[1:].lstrip('0') or '0')
                                    col_name = f"Q{num}"
                                except:
                                    col_name = f"Q{key}" if not key.startswith('Q') else key
                            else:
                                col_name = f"Q{key}" if not key.startswith('Q') else key
                            
                            if col_name not in df.columns:
                                df.at[idx, col_name] = value
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    # 파싱 실패는 무시 (로그는 verbose 모드에서만)
                    pass
    
    return df


def _create_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """기본 피처 생성"""
    # age 처리
    if 'age_raw' in df.columns:
        df['age'] = pd.to_numeric(df['age_raw'], errors='coerce')
    
    # gender 이진 변수
    if 'gender' in df.columns:
        df['gender_M'] = (df['gender'] == 'M').astype(int)
    
    # 지역 이진 변수
    if 'region_lvl1' in df.columns:
        df['is_capital_area'] = df['region_lvl1'].isin(['서울', '경기']).astype(int)
        df['is_metropolitan'] = df['region_lvl1'].isin(['서울', '부산', '대구', '인천', '광주', '대전', '울산']).astype(int)
    
    # 소득 처리
    if 'income_personal' in df.columns:
        df['income_personal'] = pd.to_numeric(df['income_personal'], errors='coerce')
    if 'income_household' in df.columns:
        df['income_household'] = pd.to_numeric(df['income_household'], errors='coerce')
    
    return df


def _create_scaled_features(df: pd.DataFrame) -> pd.DataFrame:
    """스케일링된 피처 생성"""
    # age_z (표준화) 및 age_scaled (MinMax 정규화)
    if 'age' in df.columns:
        age_values = df['age'].dropna()
        if len(age_values) > 0:
            # Z-score 정규화
            scaler_z = StandardScaler()
            df['age_z'] = pd.Series(
                scaler_z.fit_transform(age_values.values.reshape(-1, 1)).flatten(),
                index=age_values.index
            )
            # MinMax 정규화 (0~1 범위)
            scaler_mm = MinMaxScaler()
            df['age_scaled'] = pd.Series(
                scaler_mm.fit_transform(age_values.values.reshape(-1, 1)).flatten(),
                index=age_values.index
            )
    
    # Q6_scaled (소득 표준화)
    # Q6 데이터는 qa_answers에서 파싱되거나 income_personal/household에서 가져올 수 있음
    income_col = None
    income_values = None
    
    # 1순위: Q6 컬럼이 직접 있는 경우 (qa_answers에서 파싱된 경우)
    if 'Q6' in df.columns:
        income_values = pd.to_numeric(df['Q6'], errors='coerce').dropna()
    # 2순위: income_personal
    elif 'income_personal' in df.columns:
        income_col = 'income_personal'
        income_values = pd.to_numeric(df[income_col], errors='coerce').dropna()
    # 3순위: income_household
    elif 'income_household' in df.columns:
        income_col = 'income_household'
        income_values = pd.to_numeric(df[income_col], errors='coerce').dropna()
    
    if income_values is not None and len(income_values) > 0:
        scaler = StandardScaler()
        df['Q6_scaled'] = pd.Series(
            scaler.fit_transform(income_values.values.reshape(-1, 1)).flatten(),
            index=income_values.index
        )
    
    # education_level_scaled (학력 표준화)
    # Q4 데이터는 qa_answers에서 파싱되거나 education_level에서 가져올 수 있음
    education_values = None
    
    # 1순위: education_level_scaled가 이미 있는 경우
    if 'education_level_scaled' in df.columns:
        pass  # 이미 있음
    # 2순위: Q4 컬럼이 있는 경우 (qa_answers에서 파싱된 경우)
    elif 'Q4' in df.columns:
        education_values = pd.to_numeric(df['Q4'], errors='coerce').dropna()
    # 3순위: education_level 컬럼이 있는 경우
    elif 'education_level' in df.columns:
        # education_level을 숫자로 변환 (1=고졸이하, 2=대학재학, 3=대졸, 4=대학원)
        edu_map = {
            '고졸 이하': 1, '고등학교 졸업 이하': 1,
            '대학 재학': 2, '대학교 재학(휴학 포함)': 2,
            '대졸': 3, '대학교 졸업': 3,
            '대학원': 4, '대학원 이상': 4
        }
        edu_numeric = df['education_level'].map(edu_map).fillna(0)
        education_values = edu_numeric[edu_numeric > 0]
    
    if education_values is not None and len(education_values) > 0:
        # MinMax 정규화 (0~1 범위)
        scaler_mm = MinMaxScaler()
        df['education_level_scaled'] = pd.Series(
            scaler_mm.fit_transform(education_values.values.reshape(-1, 1)).flatten(),
            index=education_values.index
        )
    
    # Q8 관련 피처 계산 (w2_data에서 Q8 파싱)
    # Q8은 전자제품 리스트 (예: [1, 3, 5, 9])
    # 카테고리: kitchen(1-7), cleaning(8-14), computing(15-21), comfort(22-28)
    # 프리미엄 제품: 3, 9, 18, 20, 22, 25 등
    
    # Q8 데이터 파싱 및 계산
    if 'Q8' in df.columns:
        # Q8 리스트를 파싱하여 피쳐 계산
        q8_counts = []
        q8_kitchen_counts = []
        q8_cleaning_counts = []
        q8_computing_counts = []
        q8_comfort_counts = []
        q8_premium_indices = []
        
        for idx, q8_value in df['Q8'].items():
            # Q8 값 파싱 (리스트 또는 문자열)
            q8_list = []
            if pd.notna(q8_value):
                try:
                    if isinstance(q8_value, str):
                        # JSON 문자열 파싱
                        import json
                        q8_list = json.loads(q8_value) if q8_value.startswith('[') else []
                    elif isinstance(q8_value, list):
                        q8_list = q8_value
                    elif isinstance(q8_value, (int, float)):
                        q8_list = [int(q8_value)]
                except:
                    q8_list = []
            
            # Q8_count: 총 전자제품 수
            q8_count = len(q8_list)
            q8_counts.append(q8_count)
            
            # 카테고리별 카운트
            kitchen_count = sum(1 for x in q8_list if 1 <= x <= 7)
            cleaning_count = sum(1 for x in q8_list if 8 <= x <= 14)
            computing_count = sum(1 for x in q8_list if 15 <= x <= 21)
            comfort_count = sum(1 for x in q8_list if 22 <= x <= 28)
            
            q8_kitchen_counts.append(kitchen_count)
            q8_cleaning_counts.append(cleaning_count)
            q8_computing_counts.append(computing_count)
            q8_comfort_counts.append(comfort_count)
            
            # 프리미엄 지수 계산 (프리미엄 제품 비율)
            # TODO: 새로운 프리미엄 제품 번호 [10, 13, 16, 22, 25, 26]로 변경 가능
            # 현재는 기존 번호 [3, 9, 18, 20, 22, 25] 유지 (재클러스터링 스크립트에서 별도 처리)
            premium_products = [3, 9, 18, 20, 22, 25]  # 프리미엄 제품 번호
            premium_count = sum(1 for x in q8_list if x in premium_products)
            premium_index = premium_count / max(q8_count, 1)  # 0~1 범위
            q8_premium_indices.append(premium_index)
        
        # DataFrame에 추가
        df['Q8_count'] = q8_counts
        df['Q8_cat_kitchen_count'] = q8_kitchen_counts
        df['Q8_cat_cleaning_count'] = q8_cleaning_counts
        df['Q8_cat_computing_count'] = q8_computing_counts
        df['Q8_cat_comfort_count'] = q8_comfort_counts
        df['Q8_premium_index'] = q8_premium_indices
        
        # Q8_count_scaled: Q8_count를 MinMax 정규화
        if 'Q8_count' in df.columns:
            q8_count_values = df['Q8_count'].dropna()
            if len(q8_count_values) > 0 and q8_count_values.max() > q8_count_values.min():
                scaler_mm = MinMaxScaler()
                df['Q8_count_scaled'] = pd.Series(
                    scaler_mm.fit_transform(q8_count_values.values.reshape(-1, 1)).flatten(),
                    index=q8_count_values.index
                )
            else:
                df['Q8_count_scaled'] = 0.0
    else:
        # Q8 데이터가 없으면 기본값 0으로 설정
        q8_features = [
            'Q8_count',
            'Q8_count_scaled',
            'Q8_cat_kitchen_count',
            'Q8_cat_cleaning_count',
            'Q8_cat_computing_count',
            'Q8_cat_comfort_count',
            'Q8_premium_index'
        ]
        for feat in q8_features:
            if feat not in df.columns:
                df[feat] = 0.0
    
    # 기본 이진 변수들 (없으면 0으로 설정)
    binary_features = [
        'has_children',
        'is_college_graduate',
        'has_car',
        'has_drinking_experience',
        'has_smoking_experience',
        'is_employed',
        'is_unemployed',
        'is_student',
        'Q1_미혼',
        'Q1_기혼',
        'Q1_기타',
        'family_type_미혼',
        'family_type_기혼_자녀있음',
        'family_type_기혼_자녀없음',
        'family_type_기타_자녀있음',
        'family_type_기타_자녀없음',
        'Q5_4',
        'Q5_7',
        'Q5_8',
        'Q5_12',
        'Q5_13',
        'Q5_14',
        'Q5_15',
        'Q5_16',
        'Q8_3',
        'Q8_9',
        'Q8_18',
        'Q8_20',
        'Q8_22',
        'Q8_25'
    ]
    for feat in binary_features:
        if feat not in df.columns:
            df[feat] = 0
    
    # 결측치 처리
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

