# Rangrez dashboard as a Hugging Face Space: the full dye-lab loop on the simulator,
# every dip a real-looking phone photo. Meta models run only with the live passcode.
FROM python:3.12-slim
RUN pip install --no-cache-dir uv==0.9.5
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev && chmod -R a+rX /app
ENV RANGREZ_HOST=0.0.0.0 RANGREZ_PORT_HTTP=7860 RANGREZ_RECORD=0 RANGREZ_STEP_DELAY=1.2 \
    HOME=/tmp PYTHONUNBUFFERED=1
EXPOSE 7860
CMD ["/app/.venv/bin/rangrez-lab"]
