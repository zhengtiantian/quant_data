import uvicorn
from api_server import app
from health_checker import start_health_monitor
from scheduler.task import main_loop as start_scheduler
import threading

if __name__ == "__main__":
    # 启动健康检查后台线程
    start_health_monitor()
    
    # 启动定时任务后台线程
    threading.Thread(target=start_scheduler, daemon=True).start()

    # 启动 FastAPI 服务
    uvicorn.run(app, host="0.0.0.0", port=8000)