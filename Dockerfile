FROM python:3.11-slim

WORKDIR /app

COPY Backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY Backend/ /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
