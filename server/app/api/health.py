"""Health check API"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def ping():
    """기본 Health check"""
    return {"ok": True}


@router.get("/healthz")
async def healthz():
    """
    헬스체크 및 사전 점검
    
    Returns:
        {
            "ok": true
        }
    """
    return {
        "ok": True
    }
