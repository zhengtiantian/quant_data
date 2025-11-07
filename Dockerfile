# ========== 构建阶段 ==========
FROM python:3.12-slim AS builder

WORKDIR /app

# 拷贝依赖文件
COPY requirements.txt .

# 安装依赖（使用国内镜像可选：--index-url https://pypi.tuna.tsinghua.edu.cn/simple）
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目所有源代码
COPY . .

# ========== 运行阶段 ==========
FROM python:3.12-slim

WORKDIR /app

# 从构建阶段复制 Python 环境与项目
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app .

# 健康检查：数据库连接
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s \
  CMD python test_db_connection.py || exit 1

# 默认启动命令
CMD ["python", "main.py"]