FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt patches/constraints.txt ./
COPY converters/requirements.txt ./converters-requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -c constraints.txt -r requirements.txt && \
    pip3 install --no-cache-dir -r converters-requirements.txt

COPY patches/patch_motmetrics.py /tmp/patch_motmetrics.py
RUN python3 /tmp/patch_motmetrics.py && rm /tmp/patch_motmetrics.py

COPY . .

EXPOSE 8050

CMD ["bash"]
