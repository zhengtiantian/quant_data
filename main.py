import uvicorn
from api_server import app
from health_checker import start_health_monitor
from scheduler.task import main_loop as start_scheduler
import threading

if __name__ == "__main__":
    # Start health check background thread
    start_health_monitor()

    # Start scheduled task background thread
    threading.Thread(target=start_scheduler, daemon=True).start()

    # Start FastAPI service
    uvicorn.run(app, host="0.0.0.0", port=8000)