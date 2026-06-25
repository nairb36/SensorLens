FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

WORKDIR /app

COPY demo/requirements.txt ./requirements.txt
COPY patches/constraints.txt ./constraints.txt
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt

COPY patches/patch_motmetrics.py /tmp/patch_motmetrics.py
RUN python /tmp/patch_motmetrics.py && rm /tmp/patch_motmetrics.py

COPY --chown=user sensorlens/ ./sensorlens/
COPY --chown=user demo/ ./demo/

USER user

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "demo.app:server"]
