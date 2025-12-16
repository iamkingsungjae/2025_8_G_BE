"""서버 실행 스크립트 (Windows 이벤트 루프 정책 설정 포함)"""
import sys
import asyncio
from pathlib import Path

# Windows에서 ProactorEventLoop 문제 해결 (uvicorn 시작 전에 설정)
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    print("[BOOT] Windows 이벤트 루프 정책 설정: WindowsSelectorEventLoopPolicy")

# uvicorn 실행
if __name__ == "__main__":
    import uvicorn
    
    # 현재 스크립트가 있는 디렉토리로 이동 (server 디렉토리)
    script_dir = Path(__file__).parent.resolve()
    import os
    os.chdir(script_dir)
    
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8004,
        reload=False,  # reload는 이벤트 루프 정책과 충돌할 수 있음
        log_level="info"
    )

