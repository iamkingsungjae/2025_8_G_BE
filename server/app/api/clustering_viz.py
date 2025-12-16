"""
클러스터링 시각화 데이터 API
프론트엔드에서 recharts로 시각화하기 위한 데이터 제공
"""
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import pandas as pd
import pandas.api.types as pd_types
import numpy as np

from app.clustering.artifacts import load_artifacts

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/clustering/viz", tags=["clustering-viz"])


# 프로파일용 피쳐 세트 정의
PROFILE_FEATURES = {
    "demographic": [
        "age", "age_group", "generation",
        "family_type", "has_children", "children_category",
        "region_category", "is_metro", "is_metro_city",
    ],
    "economic": [
        "Q6_income", "Q6_scaled", "Q6_category",
        "is_employed", "is_unemployed", "is_student",
    ],
    "device_premium": [
        "Q8_count", "Q8_count_scaled",
        "Q8_premium_index", "Q8_premium_count",
        "is_apple_user", "is_samsung_user", "is_premium_phone",
        "has_car", "is_premium_car", "is_domestic_car",
    ],
    "lifestyle": [
        "has_drinking_experience", "drinking_types_count",
        "drinks_beer", "drinks_soju", "drinks_wine", "drinks_western",
        "drinks_makgeolli", "drinks_low_alcohol", "drinks_cocktail",
        "has_smoking_experience", "smoking_types_count",
        "smokes_regular", "smokes_heet", "smokes_liquid", "smokes_other",
    ],
}

# 효과 크기 임계값
EFFECT_THRESHOLDS = {
    "numeric": 0.4,
    "binary": 0.2,
}


def summarize_feature(df: pd.DataFrame, col: str) -> Optional[dict]:
    """전체 df 및 각 클러스터 df에 대해 feature별 요약 통계를 계산"""
    if col not in df.columns:
        return None
    
    s = df[col].dropna()
    if s.empty:
        return None
    
    # 이진 (0/1 또는 bool)
    if pd_types.is_bool_dtype(s) or s.dropna().isin([0, 1]).all():
        return {"type": "binary", "p": float(s.mean()), "n": int(s.count())}
    
    # 숫자형
    if pd_types.is_numeric_dtype(s):
        return {
            "type": "numeric",
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0) or 0.0),
            "median": float(s.median()),
            "n": int(s.count()),
        }
    
    # 범주형
    vc = s.value_counts(normalize=True).head(5)
    return {
        "type": "categorical",
        "top": [{"value": idx, "p": float(p)} for idx, p in vc.items()],
        "n": int(s.count()),
    }


def get_visual_strength(effect_size: float) -> str:
    """시각적 강도 표현 (⚡⚡⚡⚡⚡ ~ ⚡)"""
    abs_es = abs(effect_size)
    if abs_es > 1.0:
        return "⚡⚡⚡⚡⚡"
    elif abs_es >= 0.8:
        return "⚡⚡⚡⚡"
    elif abs_es >= 0.5:
        return "⚡⚡⚡"
    elif abs_es >= 0.3:
        return "⚡⚡"
    elif abs_es >= 0.2:
        return "⚡"
    else:
        return ""

def get_visual_bar(effect_size: float) -> str:
    """프로그레스바 표현 (10단계)"""
    abs_es = min(abs(effect_size), 1.5)  # 최대 1.5로 제한
    filled = int(abs_es / 1.5 * 10)
    return "█" * filled + "░" * (10 - filled)

def get_user_friendly_message(
    feature: str,
    cluster_mean: float,
    overall_mean: float,
    diff: float,
    effect_size: float,
    feature_labels: Optional[Dict[str, str]] = None
) -> str:
    """사용자 친화적 메시지 생성"""
    if feature_labels is None:
        feature_labels = {}
    
    feature_label = feature_labels.get(feature, feature)
    
    if feature == "age":
        if diff < 0:
            return f"이 그룹은 평균보다 {abs(diff):.1f}년 이상 젊어요"
        else:
            return f"이 그룹은 평균보다 {diff:.1f}년 이상 나이가 많아요"
    elif "income" in feature.lower():
        if diff > 0:
            return f"이 그룹의 평균 소득이 {diff:.0f}만원 더 높아요"
        else:
            return f"이 그룹의 평균 소득이 {abs(diff):.0f}만원 더 낮아요"
    else:
        if abs(effect_size) >= 0.5:
            strength = "크게"
        elif abs(effect_size) >= 0.3:
            strength = "상당히"
        else:
            strength = "약간"
        
        if diff > 0:
            return f"이 그룹의 {feature_label}이(가) {strength} 높아요"
        else:
            return f"이 그룹의 {feature_label}이(가) {strength} 낮아요"

def numeric_effect(cluster_stat: dict, overall_stat: dict) -> Optional[dict]:
    """클러스터 vs 전체 간 차이를 effect size 형태로 계산"""
    if not cluster_stat or not overall_stat:
        return None
    if overall_stat.get("type") != "numeric":
        return None
    
    std = overall_stat.get("std") or 0.0
    if std == 0:
        return None
    
    cm = cluster_stat["mean"]
    om = overall_stat["mean"]
    diff = cm - om
    d = diff / std  # effect size (Cohen's d 느낌)
    
    # 시각적 표현 추가
    visual_strength = get_visual_strength(d)
    visual_bar = get_visual_bar(d)
    
    # 해석 생성
    abs_d = abs(d)
    if abs_d > 1.0:
        interpretation = "극히 높음" if d > 0 else "극히 낮음"
    elif abs_d >= 0.8:
        interpretation = "매우 높음" if d > 0 else "매우 낮음"
    elif abs_d >= 0.5:
        interpretation = "높음" if d > 0 else "낮음"
    elif abs_d >= 0.3:
        interpretation = "보통 높음" if d > 0 else "보통 낮음"
    elif abs_d >= 0.2:
        interpretation = "약간 높음" if d > 0 else "약간 낮음"
    else:
        interpretation = "비슷함"
    
    # 사용자 친화적 메시지
    user_friendly = get_user_friendly_message("", cm, om, diff, d)
    
    return {
        "type": "numeric",
        "cluster_mean": float(cm),
        "overall_mean": float(om),
        "diff": float(diff),
        "effect_size": float(d),
        "visual_strength": visual_strength,
        "visual_bar": visual_bar,
        "interpretation": interpretation,
        "user_friendly": user_friendly,
    }


def binary_effect(cluster_stat: dict, overall_stat: dict, min_p: float = 0.05) -> Optional[dict]:
    """클러스터 vs 전체 간 차이를 penetration index 형태로 계산"""
    if not cluster_stat or not overall_stat:
        return None
    if overall_stat.get("type") != "binary":
        return None
    
    p_c = float(cluster_stat["p"])
    p_o = float(overall_stat["p"])
    if p_o < min_p:
        # 전체에서 너무 희귀하면 효과 계산 스킵
        return None
    
    index = p_c / p_o if p_o > 0 else 0.0
    lift = index - 1.0
    
    # 시각적 표현 추가
    visual_strength = get_visual_strength(abs(lift))
    visual_bar = get_visual_bar(abs(lift))
    
    # 해석 생성
    abs_lift = abs(lift)
    if abs_lift >= 0.5:
        interpretation = "매우 높음" if lift > 0 else "매우 낮음"
    elif abs_lift >= 0.3:
        interpretation = "높음" if lift > 0 else "낮음"
    elif abs_lift >= 0.2:
        interpretation = "보통 높음" if lift > 0 else "보통 낮음"
    else:
        interpretation = "비슷함"
    
    # 사용자 친화적 메시지
    diff_pct = (p_c - p_o) * 100
    if lift > 0:
        user_friendly = f"이 그룹의 {p_c*100:.0f}%가 해당 특성을 가지고 있어요 (전체 평균의 {index:.1f}배)"
    else:
        user_friendly = f"이 그룹의 {p_c*100:.0f}%가 해당 특성을 가지고 있어요 (전체 평균보다 {abs(diff_pct):.0f}%p 낮음)"
    
    return {
        "type": "binary",
        "cluster_p": p_c,
        "overall_p": p_o,
        "index": float(index),
        "lift": float(lift),
        "visual_strength": visual_strength,
        "visual_bar": visual_bar,
        "interpretation": interpretation,
        "user_friendly": user_friendly,
    }


def collect_balanced_distinctive_features(
    df: pd.DataFrame,
    cluster_id: int,
    profile_features: dict,
    overall_stats: dict,
    max_features: int = 10
) -> Tuple[List[dict], Dict[str, dict]]:
    """균형 잡힌 특징 피처 수집 (카테고리별 할당량 보장)"""
    cluster_df = df[df["cluster"] == cluster_id]
    cluster_stats: Dict[str, dict] = {}
    
    # 카테고리별 할당량
    allocation = {
        "demographic": 3,
        "economic": 2,
        "device_premium": 2,
        "lifestyle": 2,
        "family": 1  # family는 demographic에 포함될 수 있음
    }
    
    # 각 프로파일 feature에 대한 클러스터 요약 통계 계산
    for group_cols in profile_features.values():
        for col in group_cols:
            if col not in df.columns:
                continue
            if col not in cluster_stats:
                cluster_stats[col] = summarize_feature(cluster_df, col)
    
    results_by_group: Dict[str, List[dict]] = {g: [] for g in profile_features.keys()}
    
    for group, cols in profile_features.items():
        for col in cols:
            if col not in df.columns:
                continue
            
            c_stat = cluster_stats.get(col)
            o_stat = overall_stats.get(col)
            if not c_stat or not o_stat:
                continue
            
            if c_stat["type"] == "numeric" and o_stat["type"] == "numeric":
                eff = numeric_effect(c_stat, o_stat)
                if not eff:
                    continue
                if abs(eff["effect_size"]) < EFFECT_THRESHOLDS["numeric"]:
                    continue
                score = abs(eff["effect_size"])
                eff_type = "numeric"
            elif c_stat["type"] == "binary" and o_stat["type"] == "binary":
                eff = binary_effect(c_stat, o_stat)
                if not eff:
                    continue
                if abs(eff["lift"]) < EFFECT_THRESHOLDS["binary"]:
                    continue
                score = abs(eff["lift"])
                eff_type = "binary"
            else:
                continue
            
            results_by_group[group].append({
                "feature": col,
                "group": group,
                "type": eff_type,
                "effect": eff,
                "score": float(score),
            })
    
    # 1단계: 각 카테고리에서 최소 할당량만큼 선택
    distinctive: List[dict] = []
    selected_features = set()
    
    for group, items in results_by_group.items():
        items.sort(key=lambda x: x["score"], reverse=True)
        min_count = allocation.get(group, 0)
        for item in items[:min_count]:
            if item["feature"] not in selected_features:
                distinctive.append(item)
                selected_features.add(item["feature"])
    
    # 2단계: 남은 자리는 전체에서 effect size 순으로 채우기
    remaining_slots = max_features - len(distinctive)
    if remaining_slots > 0:
        all_remaining = []
        for group, items in results_by_group.items():
            for item in items:
                if item["feature"] not in selected_features:
                    all_remaining.append(item)
        
        all_remaining.sort(key=lambda x: x["score"], reverse=True)
        for item in all_remaining[:remaining_slots]:
            distinctive.append(item)
            selected_features.add(item["feature"])
    
    # 최종 정렬
    distinctive.sort(key=lambda x: x["score"], reverse=True)
    
    return distinctive, cluster_stats

# 기존 함수는 호환성을 위해 유지
def collect_distinctive_features(
    df: pd.DataFrame,
    cluster_id: int,
    profile_features: dict,
    overall_stats: dict,
) -> Tuple[List[dict], Dict[str, dict]]:
    """도메인별로 특징적인 피쳐를 골라내서, 전체 상위 5개 정도만 남김 (기존 호환)"""
    return collect_balanced_distinctive_features(df, cluster_id, profile_features, overall_stats, max_features=5)


def life_stage(cluster_stats: Dict[str, dict], overall_stats: Dict[str, dict]) -> str:
    """라이프 스테이지 판단"""
    age_cs = cluster_stats.get("age")
    age_os = overall_stats.get("age")
    if age_cs and age_os and age_cs.get("type") == "numeric" and age_os.get("type") == "numeric":
        cm = age_cs["mean"]
        om = age_os["mean"]
        diff = cm - om
        if diff <= -5:
            return "젊은"
        elif diff >= 5:
            return "중장년"
        else:
            return "중간 연령"
    return "일반"


def value_level(distinctive: List[dict]) -> str:
    """소득 레벨 판단"""
    income_eff = next(
        (
            d
            for d in distinctive
            if d["feature"] in ("Q6_income", "Q6_scaled")
            and d["effect"].get("type") == "numeric"
        ),
        None,
    )
    if not income_eff:
        return "실속형"
    
    d = income_eff["effect"]["effect_size"]
    if d >= 0.7:
        return "고소득"
    if d <= -0.7:
        return "저소득"
    return "중간 소득"


def flavor_tag(distinctive: List[dict]) -> str:
    """프리미엄/라이프스타일 태그"""
    by_feature = {d["feature"]: d for d in distinctive}
    
    def get_lift(feat: str) -> float:
        eff = by_feature.get(feat, {}).get("effect")
        if not eff or eff.get("type") != "binary":
            return 0.0
        return float(eff.get("lift") or 0.0)
    
    def get_d(feat: str) -> float:
        eff = by_feature.get(feat, {}).get("effect")
        if not eff or eff.get("type") != "numeric":
            return 0.0
        return float(eff.get("effect_size") or 0.0)
    
    # 프리미엄 소비
    if get_d("Q8_premium_index") > 0.5 or get_lift("is_premium_car") > 0.3:
        return "프리미엄 소비"
    
    # 테크 프리미엄
    if get_lift("is_apple_user") > 0.3 and get_lift("is_premium_phone") > 0.3:
        return "테크 프리미엄"
    
    # 와인·양주 선호
    if get_lift("drinks_wine") > 0.3 or get_lift("drinks_western") > 0.3:
        return "와인·양주 선호"
    
    # 건강 지향 (흡연/음주 모두 낮음)
    if get_lift("has_smoking_experience") < -0.3 and get_lift("has_drinking_experience") < -0.3:
        return "건강 지향"
    
    return ""


def build_two_tier_cluster_name(
    cluster_id: int,
    distinctive: List[dict],
    cluster_stats: Dict[str, dict],
    overall_stats: Dict[str, dict],
) -> Dict[str, str]:
    """
    2단계 군집명 생성 (실제 데이터 기반 동적 생성)
    - 메인 이름: 짧고 임팩트 있게 (실제 데이터 기반)
    - 서브 설명: 상세 특징 (실제 데이터 기반)
    """
    main_parts = []
    sub_parts = []
    
    # === 메인 이름 생성 (최대 3-4단어) ===
    
    # 1. 연령대 (가장 중요한 인구통계, 먼저 결정)
    age_eff = next(
        (d for d in distinctive 
         if d["feature"] == "age" 
         and d["effect"].get("type") == "numeric"),
        None
    )
    age_mean = None
    if age_eff:
        age_mean = age_eff["effect"]["cluster_mean"]
    else:
        age_cs = cluster_stats.get("age")
        if age_cs and age_cs.get("type") == "numeric":
            age_mean = age_cs["mean"]
    
    if age_mean is not None:
        if age_mean < 30:
            main_parts.append("20대")
        elif age_mean < 40:
            main_parts.append("30대")
        elif age_mean < 50:
            main_parts.append("40대")
        elif age_mean < 60:
            main_parts.append("50대")
        else:
            main_parts.append("60대")
    
    # 2. 소득 레벨 (메인 이름에 우선 반영)
    income_eff = next(
        (d for d in distinctive 
         if d["feature"] in ("Q6_income", "Q6_scaled") 
         and d["effect"].get("type") == "numeric"),
        None
    )
    income_effect_size = 0.0
    income_mean = None
    income_overall_mean = None
    if income_eff:
        income_effect_size = income_eff["effect"]["effect_size"]
        income_mean = income_eff["effect"].get("cluster_mean")
        income_overall_mean = income_eff["effect"].get("overall_mean")
        if income_effect_size >= 0.7:
            main_parts.append("고소득")
            sub_parts.append("고소득")
        elif income_effect_size >= 0.3:
            sub_parts.append("중상소득")
        elif income_effect_size <= -0.7:
            main_parts.append("저소득")
            sub_parts.append("저소득")
        elif income_effect_size <= -0.3:
            main_parts.append("중하소득")
            sub_parts.append("중하소득")
    else:
        # distinctive에 없으면 cluster_stats에서 직접 확인
        income_cs = cluster_stats.get("Q6_income") or cluster_stats.get("Q6_scaled")
        income_os = overall_stats.get("Q6_income") or overall_stats.get("Q6_scaled")
        if income_cs and income_os and income_cs.get("type") == "numeric" and income_os.get("type") == "numeric":
            income_mean = income_cs["mean"]
            income_overall_mean = income_os["mean"]
            std_o = income_os.get("std", 1.0) or 1.0
            income_effect_size = (income_mean - income_overall_mean) / std_o if std_o > 0 else 0.0
            if income_effect_size >= 0.7:
                main_parts.append("고소득")
                sub_parts.append("고소득")
            elif income_effect_size >= 0.3:
                sub_parts.append("중상소득")
            elif income_effect_size <= -0.7:
                main_parts.append("저소득")
                sub_parts.append("저소득")
            elif income_effect_size <= -0.3:
                main_parts.append("중하소득")
                sub_parts.append("중하소득")
    
    # 3. 프리미엄/실용 성향 (프리미엄차 우선 체크)
    car_eff = next(
        (d for d in distinctive 
         if d["feature"] == "is_premium_car"
         and d["effect"].get("type") == "binary"
         and d["effect"].get("lift", 0) > 0.3),
        None
    )
    if car_eff:
        main_parts.append("프리미엄차")
    else:
        # 프리미엄 지수 체크
        premium_eff = next(
            (d for d in distinctive 
             if d["feature"] in ("Q8_premium_index", "is_apple_user")
             and d["effect"].get("type") in ("numeric", "binary")),
            None
        )
        if premium_eff:
            eff = premium_eff["effect"]
            if eff.get("type") == "numeric" and eff.get("effect_size", 0) > 0.5:
                main_parts.append("프리미엄")
            elif eff.get("type") == "binary" and eff.get("lift", 0) > 0.3:
                main_parts.append("프리미엄")
            elif eff.get("type") == "numeric" and eff.get("effect_size", 0) < -0.3:
                main_parts.append("실속형")
    
    # 4. 가족 구성 (메인 이름과 서브 설명 모두에 사용)
    children_eff = next(
        (d for d in distinctive 
         if d["feature"] == "has_children"
         and d["effect"].get("type") == "binary"),
        None
    )
    has_children = None
    if children_eff:
        lift = children_eff["effect"].get("lift", 0)
        if lift > 0.2:
            has_children = True
            sub_parts.append("자녀 있는")
        elif lift < -0.2:
            has_children = False
            sub_parts.append("자녀 없는")
    else:
        # distinctive에 없으면 cluster_stats에서 직접 확인
        children_cs = cluster_stats.get("has_children")
        if children_cs and children_cs.get("type") == "binary":
            p = children_cs.get("p", 0)
            if p > 0.6:
                has_children = True
                sub_parts.append("자녀 있는")
            elif p < 0.4:
                has_children = False
                sub_parts.append("자녀 없는")
    
    # 5. 가족/싱글 (연령대 다음에 추가)
    if has_children is True and age_mean is not None:
        if age_mean >= 60:
            # 시니어는 이미 "60대"로 표시되므로 "가족"만 추가
            main_parts.append("가족")
        elif age_mean < 40:
            if "20대" in main_parts or "30대" in main_parts:
                main_parts.append("가족")
        elif age_mean >= 40 and age_mean < 60:
            if "40대" in main_parts or "50대" in main_parts:
                main_parts.append("가족")
    elif has_children is False and age_mean is not None and age_mean < 40:
        if "20대" in main_parts or "30대" in main_parts:
            main_parts.append("싱글")
    
    # 메인 이름 생성 (최대 4단어)
    main_name = " ".join(main_parts[:4]) if main_parts else f"군집 {cluster_id}"
    
    # === 서브 설명 생성 (상세 특징) ===
    # (소득, 가족 구성은 이미 위에서 추가됨)
    
    # 애플 유저 (특정 군집에만 표시)
    apple_eff = next(
        (d for d in distinctive 
         if d["feature"] == "is_apple_user"
         and d["effect"].get("type") == "binary"
         and d["effect"].get("lift", 0) > 0.3),
        None
    )
    if apple_eff and "애플유저" not in "·".join(sub_parts):
        sub_parts.append("애플유저")
    
    # 평균 연령 (구체적 수치)
    if age_mean is not None:
        sub_parts.append(f"평균 {age_mean:.0f}세")
    
    # 중복 제거 (순서 유지)
    seen = set()
    sub_parts_unique = []
    for item in sub_parts:
        if item not in seen:
            seen.add(item)
            sub_parts_unique.append(item)
    sub_parts = sub_parts_unique
    
    # 서브 설명 생성
    sub_description = "·".join(sub_parts) if sub_parts else ""
    
    return {
        "main": main_name,
        "sub": sub_description
    }


def generate_hierarchical_tags(
    distinctive: List[dict],
    cluster_stats: Dict[str, dict],
    overall_stats: Dict[str, dict],
    percentage: float
) -> Dict[str, List[Dict[str, Any]]]:
    """계층적 태그 생성 (1차 + 2차 + 라이프스타일)"""
    
    # 아이콘 매핑
    ICON_MAP = {
        "premium": "💎",
        "tech": "📱",
        "age_20s": "👔",
        "age_30s": "💼",
        "age_40s": "👨‍💼",
        "age_50s": "👴",
        "metro": "🏙️",
        "education": "🎓",
        "family": "👨‍👩‍👧",
        "apple": "🍎",
        "wine": "🍷",
        "health": "💪"
    }
    
    primary_tags = []
    secondary_tags = []
    lifestyle_tags = []
    
    # 1차 태그: Effect Size가 가장 큰 특징 1개
    if distinctive:
        top_feature = max(distinctive, key=lambda x: x["score"])
        feature = top_feature["feature"]
        if "premium" in feature.lower() or "apple" in feature.lower():
            primary_tags.append({
                "label": "프리미엄",
                "icon": ICON_MAP.get("premium", "💎"),
                "color": "purple",
                "category": "consumption"
            })
        elif "income" in feature.lower():
            income_eff = top_feature["effect"]
            if income_eff.get("type") == "numeric" and income_eff.get("effect_size", 0) >= 0.7:
                primary_tags.append({
                    "label": "고소득",
                    "icon": "💰",
                    "color": "gold",
                    "category": "economic"
                })
    
    # 연령대 (항상 포함)
    age_eff = next(
        (d for d in distinctive 
         if d["feature"] == "age" 
         and d["effect"].get("type") == "numeric"),
        None
    )
    if age_eff:
        cm = age_eff["effect"]["cluster_mean"]
        if cm < 30:
            primary_tags.append({
                "label": "20대",
                "icon": ICON_MAP["age_20s"],
                "color": "blue",
                "category": "demographic"
            })
        elif cm < 40:
            primary_tags.append({
                "label": "30대",
                "icon": ICON_MAP["age_30s"],
                "color": "blue",
                "category": "demographic"
            })
        elif cm < 50:
            primary_tags.append({
                "label": "40대",
                "icon": ICON_MAP["age_40s"],
                "color": "blue",
                "category": "demographic"
            })
        elif cm < 60:
            primary_tags.append({
                "label": "50대",
                "icon": ICON_MAP["age_50s"],
                "color": "blue",
                "category": "demographic"
            })
    else:
        age_cs = cluster_stats.get("age")
        if age_cs and age_cs.get("type") == "numeric":
            cm = age_cs["mean"]
            if cm < 30:
                primary_tags.append({
                    "label": "20대",
                    "icon": ICON_MAP["age_20s"],
                    "color": "blue",
                    "category": "demographic"
                })
            elif cm < 40:
                primary_tags.append({
                    "label": "30대",
                    "icon": ICON_MAP["age_30s"],
                    "color": "blue",
                    "category": "demographic"
                })
            elif cm < 50:
                primary_tags.append({
                    "label": "40대",
                    "icon": ICON_MAP["age_40s"],
                    "color": "blue",
                    "category": "demographic"
                })
    
    # 소비 성향 또는 지역 (선택, 최대 4개까지만)
    if len(primary_tags) < 4:
        premium_eff = next(
            (d for d in distinctive 
             if d["feature"] in ("Q8_premium_index", "is_premium_car")
             and d["effect"].get("type") in ("numeric", "binary")),
            None
        )
        if premium_eff and "프리미엄" not in [t["label"] for t in primary_tags]:
            primary_tags.append({
                "label": "프리미엄",
                "icon": ICON_MAP["premium"],
                "color": "purple",
                "category": "consumption"
            })
        elif len(primary_tags) < 4:
            metro_eff = next(
                (d for d in distinctive 
                 if d["feature"] == "is_metro"
                 and d["effect"].get("type") == "binary"
                 and d["effect"].get("lift", 0) > 0.2),
                None
            )
            if metro_eff:
                primary_tags.append({
                    "label": "도심형",
                    "icon": ICON_MAP["metro"],
                    "color": "green",
                    "category": "location"
                })
    
    # 2차 태그: 1차에 포함되지 않은 특징 중 Effect Size 상위
    used_features = {t.get("label") for t in primary_tags}
    remaining = [d for d in distinctive if d["feature"] not in used_features]
    remaining.sort(key=lambda x: x["score"], reverse=True)
    
    for d in remaining[:6]:  # 최대 6개
        feature = d["feature"]
        if "education" in feature.lower() or "college" in feature.lower():
            secondary_tags.append({
                "label": "고학력",
                "icon": ICON_MAP["education"],
                "category": "education"
            })
        elif feature == "has_children":
            secondary_tags.append({
                "label": "자녀有",
                "icon": ICON_MAP["family"],
                "category": "family"
            })
        elif "apple" in feature.lower():
            secondary_tags.append({
                "label": "애플유저",
                "icon": ICON_MAP["apple"],
                "category": "device"
            })
    
    # 라이프스타일 태그: 흡연/음주 관련만
    wine_eff = next(
        (d for d in distinctive 
         if d["feature"] in ("drinks_wine", "drinks_western")
         and d["effect"].get("type") == "binary"
         and d["effect"].get("lift", 0) > 0.3),
        None
    )
    if wine_eff:
        lifestyle_tags.append({
            "label": "와인",
            "icon": ICON_MAP["wine"],
            "category": "drinking"
        })
    
    smoke_eff = next(
        (d for d in distinctive 
         if d["feature"] == "has_smoking_experience"
         and d["effect"].get("type") == "binary"
         and d["effect"].get("lift", 0) < -0.3),
        None
    )
    if smoke_eff:
        lifestyle_tags.append({
            "label": "헬스",
            "icon": ICON_MAP["health"],
            "category": "health"
        })
    
    return {
        "primary": primary_tags[:4],  # 최대 4개
        "secondary": secondary_tags[:6],  # 최대 6개
        "lifestyle": lifestyle_tags
    }


# 기존 함수는 호환성을 위해 유지
def build_cluster_name(
    cluster_id: int,
    distinctive: List[dict],
    cluster_stats: Dict[str, dict],
    overall_stats: Dict[str, dict],
) -> str:
    """군집 이름 자동 생성 (기존 호환용)"""
    name_dict = build_two_tier_cluster_name(cluster_id, distinctive, cluster_stats, overall_stats)
    if name_dict["sub"]:
        return f"{name_dict['main']} ({name_dict['sub']})"
    return name_dict["main"]


def build_storytelling_insights(
    cluster_id: int,
    df: pd.DataFrame,
    distinctive: List[dict],
    cluster_stats: Dict[str, dict],
    overall_stats: Dict[str, dict],
    all_cluster_stats: Optional[Dict[int, Dict[str, dict]]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    스토리텔링 형식 인사이트 생성
    - Who: 이 그룹은 누구인가?
    - Why: 왜 이 그룹인가?
    - What: 무엇을 특징으로 하는가?
    - How Different: 다른 군집과 어떻게 다른가?
    """
    insights: Dict[str, List[Dict[str, Any]]] = {
        "who": [],
        "why": [],
        "what": [],
        "how_different": [],
    }
    
    cluster_df = df[df["cluster"] == cluster_id]
    size = len(cluster_df)
    total = len(df)
    pct = (size / total * 100.0) if total > 0 else 0.0
    
    # helper: distinctive에서 feature별 effect 가져오기
    by_feature = {d["feature"]: d for d in distinctive}
    
    def get_numeric_eff(name: str) -> Optional[dict]:
        d = by_feature.get(name)
        if not d:
            return None
        eff = d.get("effect")
        if eff and eff.get("type") == "numeric":
            return eff
        return None
    
    def get_binary_eff(name: str) -> Optional[dict]:
        d = by_feature.get(name)
        if not d:
            return None
        eff = d.get("effect")
        if eff and eff.get("type") == "binary":
            return eff
        return None
    
    # === Who: 이 그룹은 누구인가? ===
    if pct >= 30:
        insights["who"].append({
            "message": f"💎 이 그룹은 전체의 {pct:.1f}%를 차지하는 대형 군집이에요 ({size:,}명)",
            "strength": "⚡⚡⚡",
            "category": "size"
        })
    elif pct >= 15:
        insights["who"].append({
            "message": f"📊 이 그룹은 전체의 {pct:.1f}%를 차지하는 중형 군집이에요 ({size:,}명)",
            "strength": "⚡⚡",
            "category": "size"
        })
    else:
        insights["who"].append({
            "message": f"🔍 이 그룹은 전체의 {pct:.1f}%를 차지하는 소형 군집이에요 ({size:,}명)",
            "strength": "⚡",
            "category": "size"
        })
    
    # 연령 (distinctive에서 가져오거나, 없으면 cluster_stats에서 직접 계산)
    age_eff = get_numeric_eff("age")
    age_cs = None
    age_os = None
    
    if age_eff:
        cm = age_eff["cluster_mean"]
        om = age_eff["overall_mean"]
    else:
        # distinctive에 age가 없으면 cluster_stats에서 직접 가져오기
        age_cs = cluster_stats.get("age")
        age_os = overall_stats.get("age")
        if age_cs and age_os and age_cs.get("type") == "numeric" and age_os.get("type") == "numeric":
            cm = age_cs["mean"]
            om = age_os["mean"]
        else:
            cm = None
            om = None
    
    if cm is not None and om is not None:
        diff = cm - om
        if abs(diff) >= 5:
            # visual_strength 계산
            if age_eff:
                effect_size = age_eff.get("effect_size", abs(diff) / 10.0)
                visual_strength = get_visual_strength(effect_size)
            elif age_os:
                std_o = age_os.get("std", 1.0) or 1.0
                effect_size = abs(diff) / std_o if std_o > 0 else abs(diff) / 10.0
                visual_strength = get_visual_strength(effect_size)
            else:
                visual_strength = "⚡⚡"
            
            if diff < 0:
                insights["who"].append({
                    "message": f"👔 이 그룹의 평균 연령은 {cm:.0f}세로, 전체 평균({om:.0f}세)보다 {abs(diff):.0f}년 이상 젊어요",
                    "strength": visual_strength,
                    "category": "demographic"
                })
            else:
                insights["who"].append({
                    "message": f"👴 이 그룹의 평균 연령은 {cm:.0f}세로, 전체 평균({om:.0f}세)보다 {diff:.0f}년 이상 많아요",
                    "strength": visual_strength,
                    "category": "demographic"
                })
    
    # === Why: 왜 이 그룹인가? ===
    # 소득 (distinctive에서 가져오거나, 없으면 cluster_stats에서 직접 계산)
    income_eff = get_numeric_eff("Q6_income") or get_numeric_eff("Q6_scaled")
    if income_eff:
        cm = income_eff["cluster_mean"]
        om = income_eff["overall_mean"]
        d = income_eff["effect_size"]
    else:
        # distinctive에 income이 없으면 cluster_stats에서 직접 가져오기
        income_cs = cluster_stats.get("Q6_income") or cluster_stats.get("Q6_scaled")
        income_os = overall_stats.get("Q6_income") or overall_stats.get("Q6_scaled")
        if income_cs and income_os and income_cs.get("type") == "numeric" and income_os.get("type") == "numeric":
            cm = income_cs["mean"]
            om = income_os["mean"]
            # effect_size 계산
            std_o = income_os.get("std", 1.0) or 1.0
            d = (cm - om) / std_o if std_o > 0 else 0.0
        else:
            cm = None
            om = None
            d = 0.0
    
    if cm is not None and om is not None and abs(d) >= 0.4:
        visual_strength = get_visual_strength(abs(d)) if income_eff else "⚡⚡"
        if d > 0:
            insights["why"].append({
                "message": f"💰 이 그룹의 평균 소득은 {cm:.0f}만원으로, 전체 평균({om:.0f}만원)보다 {cm-om:.0f}만원 더 높아요",
                "strength": visual_strength,
                "category": "economic"
            })
        else:
            insights["why"].append({
                "message": f"💸 이 그룹의 평균 소득은 {cm:.0f}만원으로, 전체 평균({om:.0f}만원)보다 {om-cm:.0f}만원 더 낮아요",
                "strength": visual_strength,
                "category": "economic"
            })
    
    # === What: 무엇을 특징으로 하는가? ===
    # 프리미엄/디바이스
    premium_eff = get_numeric_eff("Q8_premium_index")
    if premium_eff and premium_eff["effect_size"] > 0.4:
        insights["what"].append({
            "message": "💎 이 그룹은 프리미엄 가전/디바이스 보유 수준이 전체보다 높아요",
            "strength": premium_eff.get("visual_strength", "⚡⚡⚡"),
            "category": "device_premium"
        })
    
    apple_eff = get_binary_eff("is_apple_user")
    if apple_eff and apple_eff["lift"] > 0.3:
        cluster_p = apple_eff["cluster_p"]
        overall_p = apple_eff["overall_p"]
        index = apple_eff["index"]
        insights["what"].append({
            "message": f"🍎 이 그룹의 절반 이상이 아이폰을 쓰고 있어요 (전체 평균의 {index:.1f}배)",
            "strength": apple_eff.get("visual_strength", "⚡⚡⚡"),
            "category": "device_premium"
        })
    
    phone_eff = get_binary_eff("is_premium_phone")
    if phone_eff and phone_eff["lift"] > 0.3:
        insights["what"].append({
            "message": "📱 이 그룹은 프리미엄 스마트폰 비율이 전체보다 높아요",
            "strength": phone_eff.get("visual_strength", "⚡⚡"),
            "category": "device_premium"
        })
    
    # === How Different: 다른 군집과 어떻게 다른가? ===
    # 군집 간 상대적 포지셔닝 (all_cluster_stats가 제공된 경우)
    if all_cluster_stats:
        # 연령 비교 (distinctive에서 가져오거나 cluster_stats에서 직접)
        age_eff = get_numeric_eff("age")
        if age_eff:
            cm = age_eff["cluster_mean"]
        else:
            age_cs = cluster_stats.get("age")
            if age_cs and age_cs.get("type") == "numeric":
                cm = age_cs["mean"]
            else:
                cm = None
        
        if cm is not None:
            other_ages = []
            for cid, stats in all_cluster_stats.items():
                if cid != cluster_id and stats.get("age") and stats["age"].get("type") == "numeric":
                    other_ages.append((cid, stats["age"]["mean"]))
            other_ages.sort(key=lambda x: x[1])
            
            younger_count = sum(1 for _, age in other_ages if age < cm)
            total_clusters = len(all_cluster_stats)
            position = total_clusters - younger_count
            
            if position <= total_clusters:
                insights["how_different"].append({
                    "message": f"📊 {total_clusters}개 군집 중 {position}번째로 젊은 그룹이에요",
                    "strength": "⚡⚡",
                    "category": "comparison"
                })
    
    # 프리미엄차 보유
    car_eff = get_binary_eff("is_premium_car")
    if car_eff and car_eff["lift"] > 0.3:
        cluster_p = car_eff["cluster_p"]
        overall_p = car_eff["overall_p"]
        index = car_eff["index"]
        insights["what"].append({
            "message": f"🚗 이 그룹의 프리미엄차 보유율이 {cluster_p:.1%}로, 전체 평균({overall_p:.1%})보다 {index:.1f}배 높아요",
            "strength": car_eff.get("visual_strength", "⚡⚡⚡"),
            "category": "device_premium"
        })
    
    # 자녀 유무
    children_eff = get_binary_eff("has_children")
    if children_eff:
        lift = children_eff["lift"]
        cluster_p = children_eff["cluster_p"]
        overall_p = children_eff["overall_p"]
        if lift > 0.2:
            insights["what"].append({
                "message": f"👨‍👩‍👧 이 그룹의 {cluster_p:.1%}가 자녀를 두고 있어요 (전체 평균: {overall_p:.1%})",
                "strength": children_eff.get("visual_strength", "⚡⚡"),
                "category": "demographic"
            })
        elif lift < -0.2:
            insights["what"].append({
                "message": f"👤 이 그룹의 {cluster_p:.1%}만 자녀가 있어요 (전체 평균: {overall_p:.1%})",
                "strength": children_eff.get("visual_strength", "⚡⚡"),
                "category": "demographic"
            })
    
    # 교육 수준
    education_eff = get_numeric_eff("education_level_scaled")
    if education_eff and abs(education_eff["effect_size"]) >= 0.4:
        cm = education_eff["cluster_mean"]
        om = education_eff["overall_mean"]
        if education_eff["effect_size"] > 0:
            insights["what"].append({
                "message": f"🎓 이 그룹의 평균 교육 수준이 전체보다 높아요",
                "strength": education_eff.get("visual_strength", "⚡⚡"),
                "category": "demographic"
            })
        else:
            insights["what"].append({
                "message": f"📚 이 그룹의 평균 교육 수준이 전체보다 낮아요",
                "strength": education_eff.get("visual_strength", "⚡⚡"),
                "category": "demographic"
            })
    
    # 전자제품 보유 수
    q8_count_eff = get_numeric_eff("Q8_count") or get_numeric_eff("Q8_count_scaled")
    if q8_count_eff and abs(q8_count_eff["effect_size"]) >= 0.4:
        cm = q8_count_eff["cluster_mean"]
        om = q8_count_eff["overall_mean"]
        if q8_count_eff["effect_size"] > 0:
            insights["what"].append({
                "message": f"📱 이 그룹은 평균 {cm:.1f}개의 전자제품을 보유하고 있어요 (전체 평균: {om:.1f}개)",
                "strength": q8_count_eff.get("visual_strength", "⚡⚡"),
                "category": "device_premium"
            })
        else:
            insights["what"].append({
                "message": f"📱 이 그룹은 평균 {cm:.1f}개의 전자제품을 보유하고 있어요 (전체 평균: {om:.1f}개)",
                "strength": q8_count_eff.get("visual_strength", "⚡⚡"),
                "category": "device_premium"
            })
    
    # 라이프스타일
    wine_eff = get_binary_eff("drinks_wine")
    if wine_eff and wine_eff["lift"] > 0.3:
        cluster_p = wine_eff["cluster_p"]
        index = wine_eff["index"]
        insights["what"].append({
            "message": f"🍷 이 그룹의 와인 음용 비율이 전체보다 높아요 (전체 평균의 {index:.1f}배)",
            "strength": wine_eff.get("visual_strength", "⚡⚡"),
            "category": "lifestyle"
        })
    
    smoke_eff = get_binary_eff("has_smoking_experience")
    if smoke_eff:
        lift = smoke_eff["lift"]
        if lift > 0.3:
            insights["what"].append({
                "message": "🚬 이 그룹의 흡연 경험 비율이 전체보다 높아요",
                "strength": smoke_eff.get("visual_strength", "⚡⚡"),
                "category": "lifestyle"
            })
        elif lift < -0.3:
            insights["what"].append({
                "message": "💪 이 그룹은 흡연 경험 비율이 전체보다 낮아요 (건강 지향)",
                "strength": smoke_eff.get("visual_strength", "⚡⚡"),
                "category": "lifestyle"
            })
    
    # 지역 (대도시 거주)
    metro_eff = get_binary_eff("is_metro") or get_binary_eff("is_metro_city")
    if metro_eff:
        lift = metro_eff["lift"]
        cluster_p = metro_eff["cluster_p"]
        overall_p = metro_eff["overall_p"]
        if lift > 0.2:
            insights["what"].append({
                "message": f"🏙️ 이 그룹의 {cluster_p:.1%}가 대도시에 거주해요 (전체 평균: {overall_p:.1%})",
                "strength": metro_eff.get("visual_strength", "⚡⚡"),
                "category": "demographic"
            })
        elif lift < -0.2:
            insights["what"].append({
                "message": f"🏘️ 이 그룹의 {cluster_p:.1%}가 중소도시에 거주해요 (전체 평균: {overall_p:.1%})",
                "strength": metro_eff.get("visual_strength", "⚡⚡"),
                "category": "demographic"
            })
    
    return insights


def get_cluster_positioning(
    cluster_id: int,
    feature: str,
    cluster_value: float,
    all_cluster_stats: Dict[int, Dict[str, dict]]
) -> Optional[Dict[str, Any]]:
    """
    군집 간 상대적 포지셔닝 계산
    예: "5개 군집 중 2번째로 젊은 그룹"
    """
    if not all_cluster_stats:
        return None
    
    # 모든 군집의 해당 feature 값 수집
    feature_values = []
    for cid, stats in all_cluster_stats.items():
        if feature in stats and stats[feature].get("type") == "numeric":
            feature_values.append((cid, stats[feature]["mean"]))
    
    if len(feature_values) < 2:
        return None
    
    # 정렬 (낮은 값이 좋은 경우와 높은 값이 좋은 경우 구분)
    # age의 경우 낮을수록 "젊은"이므로 역순 정렬
    if feature == "age":
        feature_values.sort(key=lambda x: x[1], reverse=True)  # 높은 값(나이 많은)이 먼저
    else:
        feature_values.sort(key=lambda x: x[1], reverse=False)  # 낮은 값이 먼저
    
    # 현재 군집의 위치 찾기
    position = None
    for idx, (cid, val) in enumerate(feature_values):
        if cid == cluster_id:
            position = idx + 1
            break
    
    if position is None:
        return None
    
    total = len(feature_values)
    
    # 포지션 설명 생성
    if feature == "age":
        if position == 1:
            description = f"{total}개 군집 중 가장 젊은 그룹"
        elif position == total:
            description = f"{total}개 군집 중 가장 나이 많은 그룹"
        else:
            description = f"{total}개 군집 중 {position}번째로 젊은 그룹"
    else:
        if position == 1:
            description = f"{total}개 군집 중 가장 낮은 그룹"
        elif position == total:
            description = f"{total}개 군집 중 가장 높은 그룹"
        else:
            description = f"{total}개 군집 중 {position}번째로 낮은 그룹"
    
    return {
        "position": position,
        "total": total,
        "description": description,
        "percentile": round((position / total) * 100, 1)
    }


def build_marketing_segments(
    cluster_id: int,
    distinctive: List[dict],
    cluster_stats: Dict[str, dict],
    overall_stats: Dict[str, dict],
    percentage: float
) -> Dict[str, Any]:
    """
    마케팅 활용 가이드 생성
    - 추천 채널
    - 제품 적합도
    - 캠페인 아이디어
    (마케팅 가치 점수는 제거됨)
    """
    segments = {}
    
    # 1. 추천 채널
    recommended_channels = []
    
    age_eff = next(
        (d for d in distinctive 
         if d["feature"] == "age"
         and d["effect"].get("type") == "numeric"),
        None
    )
    if age_eff:
        cm = age_eff["effect"]["cluster_mean"]
        if cm < 30:
            recommended_channels.extend(["인스타그램", "틱톡", "유튜브 쇼츠"])
        elif cm < 40:
            recommended_channels.extend(["유튜브", "페이스북", "네이버 블로그"])
        elif cm < 50:
            recommended_channels.extend(["네이버", "카카오톡", "이메일"])
        else:
            recommended_channels.extend(["TV 광고", "신문", "라디오"])
    
    # 프리미엄 소비자면 디지털 프리미엄 채널 추가
    premium_eff = next(
        (d for d in distinctive 
         if d["feature"] in ("Q8_premium_index", "is_premium_car", "is_apple_user")
         and d["effect"].get("type") in ("numeric", "binary")),
        None
    )
    if premium_eff:
        recommended_channels = ["유튜브 프리미엄", "넷플릭스", "디즈니+"] + recommended_channels[:3]
    
    # 2. 제품 적합도
    product_fit = []
    
    if premium_eff:
        product_fit.append({
            "category": "프리미엄 제품",
            "score": 90,
            "examples": ["명품 가방", "프리미엄 스마트폰", "고급 와인"]
        })
    
    income_eff = next(
        (d for d in distinctive 
         if d["feature"] in ("Q6_income", "Q6_scaled")
         and d["effect"].get("type") == "numeric"),
        None
    )
    if income_eff and income_eff["effect"]["effect_size"] >= 0.5:
        product_fit.append({
            "category": "고가 제품",
            "score": 85,
            "examples": ["자동차", "부동산", "투자 상품"]
        })
    
    # 3. 캠페인 아이디어
    campaign_ideas = []
    
    if age_eff and age_eff["effect"]["cluster_mean"] < 35:
        campaign_ideas.append({
            "title": "젊은 세대 타겟 캠페인",
            "concept": "트렌디하고 개성 있는 메시지",
            "hashtag": "#젊은에너지 #트렌드세터"
        })
    
    if premium_eff:
        campaign_ideas.append({
            "title": "프리미엄 라이프스타일 캠페인",
            "concept": "품질과 가치를 중시하는 메시지",
            "hashtag": "#프리미엄라이프 #품질중시"
        })
    
    segments = {
        "recommended_channels": recommended_channels[:5],
        "product_fit": product_fit,
        "campaign_ideas": campaign_ideas
    }
    
    return segments


# 기존 함수는 호환성을 위해 유지
def build_insights(
    cluster_id: int,
    df: pd.DataFrame,
    distinctive: List[dict],
    cluster_stats: Dict[str, dict],
    overall_stats: Dict[str, dict],
) -> Dict[str, List[str]]:
    """카테고리별 인사이트 생성 (기존 호환용)"""
    storytelling = build_storytelling_insights(
        cluster_id, df, distinctive, cluster_stats, overall_stats
    )
    
    # 기존 형식으로 변환
    result: Dict[str, List[str]] = {
        "size": [],
        "demographic": [],
        "economic": [],
        "device_premium": [],
        "lifestyle": [],
    }
    
    for category, items in storytelling.items():
        for item in items:
            msg = item["message"]
            if category == "who" and "size" in item.get("category", ""):
                result["size"].append(msg)
            elif category in ("who", "why") and "demographic" in item.get("category", ""):
                result["demographic"].append(msg)
            elif category == "why" and "economic" in item.get("category", ""):
                result["economic"].append(msg)
            elif category == "what" and "device_premium" in item.get("category", ""):
                result["device_premium"].append(msg)
            elif category == "what" and "lifestyle" in item.get("category", ""):
                result["lifestyle"].append(msg)
    
    return result


@router.get("/k-analysis/{session_id}")
async def get_k_analysis_data(session_id: str):
    """
    최적 K 분석 데이터 반환
    k별 Silhouette, Davies-Bouldin, Calinski-Harabasz 점수
    """
    try:
        artifacts = load_artifacts(session_id)
        if not artifacts:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        meta = artifacts.get('meta', {})
        result_meta = meta.get('result_meta', {})
        
        # k_scores가 메타데이터에 있는지 확인
        k_scores = result_meta.get('k_scores', [])
        
        if not k_scores:
            # 메타데이터에 없으면 빈 배열 반환
            return {
                'success': False,
                'message': 'K 분석 데이터가 없습니다.',
                'data': []
            }
        
        # 데이터 포맷팅
        formatted_data = []
        for score in k_scores:
            formatted_data.append({
                'k': score.get('k'),
                'silhouette': float(score.get('silhouette', 0)),
                'davies_bouldin': float(score.get('davies_bouldin', 0)),
                'calinski_harabasz': float(score.get('calinski_harabasz', 0)),
                'min_cluster_size': int(score.get('min_cluster_size', 0))
            })
        
        return {
            'success': True,
            'data': formatted_data,
            'optimal_k': result_meta.get('optimal_k')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[K 분석 데이터 오류] {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"K 분석 데이터 조회 실패: {str(e)}")


@router.get("/cluster-profiles/{session_id}")
async def get_cluster_profiles(session_id: str) -> JSONResponse:
    """
    클러스터별 피처 프로파일 데이터 반환 (v2 엔진)
    """
    logger.info(f"[클러스터 프로필 요청] session_id: {session_id}")
    
    try:
        # Precomputed 세션인 경우 precomputed API로 리다이렉트
        if session_id == 'precomputed_default':
            logger.info(f"[클러스터 프로필] Precomputed 세션 감지, precomputed API 사용")
            from app.api.precomputed import get_precomputed_profiles
            return await get_precomputed_profiles()
        
        # 1) artifacts / df / meta 로드
        logger.debug(f"[클러스터 프로필] artifacts 로드 시작: {session_id}")
        artifacts = load_artifacts(session_id)
        
        if not artifacts:
            error_msg = f"세션을 찾을 수 없습니다: {session_id}"
            logger.error(f"[클러스터 프로필 오류] {error_msg}")
            logger.debug(f"[클러스터 프로필] 세션 디렉토리 확인: runs/{session_id}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        logger.debug(f"[클러스터 프로필] artifacts 로드 완료. 키: {list(artifacts.keys())}")
        
        data = artifacts.get('data')
        if data is None:
            error_msg = "데이터를 찾을 수 없습니다."
            logger.error(f"[클러스터 프로필 오류] {error_msg}")
            logger.debug(f"[클러스터 프로필] artifacts 키: {list(artifacts.keys())}")
            raise HTTPException(status_code=404, detail=error_msg)
        
        logger.debug(f"[클러스터 프로필] 데이터 타입: {type(data)}")
        df = pd.read_csv(data) if isinstance(data, str) else data
        logger.debug(f"[클러스터 프로필] DataFrame shape: {df.shape}, 컬럼: {list(df.columns)[:10]}")
        
        if 'cluster' not in df.columns:
            error_msg = f"클러스터 정보가 없습니다. 컬럼: {list(df.columns)[:20]}"
            logger.error(f"[클러스터 프로필 오류] {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.debug(f"[클러스터 프로필] 클러스터 정보 확인 완료. 고유 클러스터: {df['cluster'].unique()[:10]}")
        
        # 메타데이터에서 사용된 피처 확인 (참고용, 프로파일에는 사용 안 함)
        meta = artifacts.get('meta', {})
        result_meta = meta.get('result_meta', {})
        algorithm_info = result_meta.get('algorithm_info', {})
        used_features = algorithm_info.get('features', [])  # 클러스터링에 사용한 피처 (참고용)
        
        # 2) 전체 stats 계산
        overall_stats: Dict[str, dict] = {}
        for group, cols in PROFILE_FEATURES.items():
            for col in cols:
                if col not in df.columns:
                    continue
                if col not in overall_stats:
                    overall_stats[col] = summarize_feature(df, col)
        
        result_clusters: List[dict] = []
        total = len(df)
        
        # 노이즈 클러스터 제외하고 처리
        valid_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        
        for cluster_id in valid_clusters:
            cluster_id_int = int(cluster_id)
            cluster_df = df[df['cluster'] == cluster_id_int]
            size = len(cluster_df)
            percentage = (size / total * 100.0) if total > 0 else 0.0
            
            # 3) 특징 피쳐 및 클러스터별 stats
            distinctive, cluster_stats = collect_distinctive_features(
                df=df,
                cluster_id=cluster_id_int,
                profile_features=PROFILE_FEATURES,
                overall_stats=overall_stats,
            )
            
            # 4) 이름/인사이트 생성
            name = build_cluster_name(
                cluster_id=cluster_id_int,
                distinctive=distinctive,
                cluster_stats=cluster_stats,
                overall_stats=overall_stats,
            )
            insights_dict = build_insights(
                cluster_id=cluster_id_int,
                df=df,
                distinctive=distinctive,
                cluster_stats=cluster_stats,
                overall_stats=overall_stats,
            )
            
            # 5) 태그: flavor_tag + size 정보 등으로 구성
            flavor = flavor_tag(distinctive)
            tags: List[str] = []
            if flavor:
                tags.append(flavor)
            if percentage >= 30:
                tags.append("대형 군집")
            elif percentage >= 15:
                tags.append("중형 군집")
            else:
                tags.append("소형 군집")
            
            # 기존 v1 호환을 위한 fields 유지
            # distinctive_features는 v2 구조를 그대로 넘기되, 기존 프론트가 기대하는 필드도 포함
            distinctive_features_v1_compat = []
            for d in distinctive:
                eff = d.get("effect", {})
                if eff.get("type") == "numeric":
                    distinctive_features_v1_compat.append({
                        "feature": d["feature"],
                        "value": eff.get("cluster_mean", 0.0),
                        "overall": eff.get("overall_mean", 0.0),
                        "diff": eff.get("diff", 0.0),
                        "diff_percent": eff.get("effect_size", 0.0) * 100,  # effect_size를 퍼센트로 변환
                    })
                elif eff.get("type") == "binary":
                    distinctive_features_v1_compat.append({
                        "feature": d["feature"],
                        "value": eff.get("cluster_p", 0.0),
                        "overall": eff.get("overall_p", 0.0),
                        "diff": eff.get("lift", 0.0),
                        "diff_percent": eff.get("lift", 0.0) * 100,
                    })
            
            # insights를 기존 형식(리스트)과 새 형식(딕셔너리) 모두 지원
            insights_list = []
            for category, items in insights_dict.items():
                insights_list.extend(items)
            
            cluster_profile = {
                "cluster": cluster_id_int,
                "size": size,
                "percentage": float(percentage),
                "name": name,
                "tags": tags,
                "distinctive_features": distinctive_features_v1_compat,  # v1 호환
                "insights": insights_list,  # v1 호환 (리스트)
                "insights_by_category": insights_dict,  # v2 새 필드 (카테고리별)
                "segments": {
                    "life_stage": life_stage(cluster_stats, overall_stats),
                    "value_level": value_level(distinctive),
                },
                # 기존 features 필드도 유지 (클러스터링에 사용한 피처 평균값)
                "features": {},
            }
            
            # 기존 features 필드 채우기 (클러스터링에 사용한 피처의 평균값)
            if used_features:
                for feat in used_features:
                    if feat in df.columns:
                        cluster_profile["features"][feat] = float(cluster_df[feat].mean())
            
            result_clusters.append(cluster_profile)
        
        response_payload = {
            "success": True,
            "data": result_clusters,
            "profile_features": PROFILE_FEATURES,
            "used_features": used_features,  # 클러스터링에 사용한 피처 (참고용)
        }
        
        return JSONResponse(content=jsonable_encoder(response_payload))
        
    except HTTPException as http_err:
        logger.error(f"[클러스터 프로필 HTTP 오류] {http_err.status_code}: {http_err.detail}")
        logger.debug(f"[클러스터 프로필] HTTP 오류 상세: session_id={session_id}")
        raise
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"[클러스터 프로필 예외 발생] {error_type}: {error_msg}", exc_info=True)
        logger.debug(f"[클러스터 프로필] 예외 발생 위치: session_id={session_id}")
        raise HTTPException(
            status_code=500, 
            detail=f"클러스터 프로파일 조회 실패: {error_type} - {error_msg}"
        )


@router.get("/cluster-distribution/{session_id}")
async def get_cluster_distribution(session_id: str):
    """
    클러스터 분포 데이터 반환 (막대그래프 + 파이차트용)
    """
    try:
        artifacts = load_artifacts(session_id)
        if not artifacts:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        data = artifacts.get('data')
        if data is None:
            raise HTTPException(status_code=404, detail="데이터를 찾을 수 없습니다.")
        
        df = pd.read_csv(data) if isinstance(data, str) else data
        
        if 'cluster' not in df.columns:
            raise HTTPException(status_code=400, detail="클러스터 정보가 없습니다.")
        
        # 클러스터별 개수 계산
        cluster_counts = df['cluster'].value_counts().sort_index()
        total = len(df)
        
        distribution_data = []
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:  # 노이즈는 별도 처리
                continue
            distribution_data.append({
                'cluster': int(cluster_id),
                'count': int(count),
                'percentage': float(count / total * 100)
            })
        
        # 노이즈가 있으면 별도 추가
        if -1 in cluster_counts.index:
            noise_count = int(cluster_counts[-1])
            distribution_data.append({
                'cluster': -1,
                'count': noise_count,
                'percentage': float(noise_count / total * 100),
                'is_noise': True
            })
        
        return {
            'success': True,
            'data': distribution_data,
            'total': int(total)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[클러스터 분포 오류] {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"클러스터 분포 조회 실패: {str(e)}")


@router.get("/correlation-matrix/{session_id}")
async def get_correlation_matrix(session_id: str):
    """
    피처 간 상관계수 매트릭스 반환
    """
    try:
        artifacts = load_artifacts(session_id)
        if not artifacts:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        data = artifacts.get('data')
        if data is None:
            raise HTTPException(status_code=404, detail="데이터를 찾을 수 없습니다.")
        
        df = pd.read_csv(data) if isinstance(data, str) else data
        
        # 메타데이터에서 사용된 피처 확인
        meta = artifacts.get('meta', {})
        result_meta = meta.get('result_meta', {})
        algorithm_info = result_meta.get('algorithm_info', {})
        
        # 사용된 피처 목록
        used_features = algorithm_info.get('features', [])
        if not used_features:
            # 숫자형 컬럼 중 cluster, mb_sn 제외
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'cluster' in numeric_cols:
                numeric_cols.remove('cluster')
            if 'mb_sn' in numeric_cols:
                numeric_cols.remove('mb_sn')
            used_features = numeric_cols[:10]  # 최대 10개
        
        # 상관계수 계산
        corr_matrix = df[used_features].corr()
        
        # JSON 직렬화 가능한 형태로 변환
        correlation_data = []
        for i, feature1 in enumerate(used_features):
            row = {'feature': feature1, 'correlations': {}}
            for j, feature2 in enumerate(used_features):
                row['correlations'][feature2] = float(corr_matrix.loc[feature1, feature2])
            correlation_data.append(row)
        
        return {
            'success': True,
            'data': correlation_data,
            'features': used_features
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[상관계수 매트릭스 오류] {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"상관계수 매트릭스 조회 실패: {str(e)}")
