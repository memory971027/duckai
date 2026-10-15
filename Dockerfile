FROM mcr.microsoft.com/playwright/python:v1.49.1-noble

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY duckai.py main.py tools.py toolrouter.py ./

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
ENV PORT=7860

EXPOSE 7860

CMD ["python", "main.py"]
