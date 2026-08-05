FROM python:3.14.6-slim-trixie

WORKDIR /backend
RUN pip install "fastapi[standard]>=0.141.1,<0.142"
COPY backend/main.py main.py
EXPOSE 8000
CMD ["fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]