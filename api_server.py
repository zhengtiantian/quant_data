from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import os
import subprocess
import signal

app = FastAPI(title="Quant Data Script API")

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 存储运行中的子进程 {脚本路径: process对象}
RUNNING_PROCESSES = {}

@app.get("/health")
def health_check():
    """健康检查"""
    return {"status": "ok"}

@app.get("/scripts")
def list_scripts():
    """递归列出所有 .py 文件"""
    script_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        if ".venv" in root or "__pycache__" in root or ".git" in root:
            continue
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                rel_path = os.path.relpath(os.path.join(root, f), BASE_DIR)
                script_files.append(rel_path)
    return {"scripts": sorted(script_files)}

@app.get("/run/{path:path}")
def run_script(path: str):
    """执行脚本并实时返回日志流"""
    script_path = os.path.join(BASE_DIR, path)
    if not os.path.exists(script_path):
        return JSONResponse({"error": f"Script not found: {path}"}, status_code=404)

    if path in RUNNING_PROCESSES:
        return JSONResponse({"error": f"Script {path} is already running."}, status_code=400)

    def stream_logs():
        process = subprocess.Popen(
            ["python", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            preexec_fn=os.setsid
        )
        RUNNING_PROCESSES[path] = process
        try:
            for line in iter(process.stdout.readline, ''):
                yield line
        finally:
            process.stdout.close()
            process.wait()
            if path in RUNNING_PROCESSES:
                del RUNNING_PROCESSES[path]

    return StreamingResponse(stream_logs(), media_type="text/plain")

@app.post("/stop/{path:path}")
def stop_script(path: str):
    """停止正在运行的脚本"""
    process = RUNNING_PROCESSES.get(path)
    if not process:
        return JSONResponse({"error": f"Script {path} is not running."}, status_code=400)
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        del RUNNING_PROCESSES[path]
        return {"message": f"Stopped {path}"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)