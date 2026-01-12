FROM python:3.12
#FROM python:3.12-slim

WORKDIR /appp

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m appuser && \
    chown -R appuser /appp && \
    chmod -R 755 /appp

USER appuser

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app

CMD ["python", "app.py"]
#CMD ["uvicorn", "main:appp", "--host", "0.0.0.0", "--port", "8080"]

# COPY requirements.txt /app/
# RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# COPY ./ /app

# CMD ["python", "main.py"]

