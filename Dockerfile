FROM python:3.10-slim AS builder

WORKDIR /build

# TA-Lib C library — HTTPS source with checksum verification
ARG TALIB_VERSION=0.6.4
ARG TALIB_SHA256=aa04066d17d69c73b1baaef0883414d3d56ab3775872d82916d1cdb376a3ae86
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget ca-certificates \
    && wget -q "https://github.com/TA-Lib/ta-lib/releases/download/v${TALIB_VERSION}/ta-lib-${TALIB_VERSION}-src.tar.gz" \
        -O ta-lib.tar.gz \
    && echo "${TALIB_SHA256}  ta-lib.tar.gz" | sha256sum -c - \
    && tar -xzf ta-lib.tar.gz \
    && cd ta-lib-${TALIB_VERSION} && ./configure --prefix=/usr && make -j"$(nproc)" && make install \
    && cd .. && rm -rf ta-lib* \
    && apt-get purge -y build-essential wget && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Runtime stage ---
FROM python:3.10-slim

WORKDIR /app

# Copy TA-Lib shared libs + Python packages from builder
COPY --from=builder /usr/lib/libta-lib* /usr/lib/
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Shared lib cache
RUN ldconfig

# Non-root user for security
RUN groupadd -r bot && useradd -r -g bot -d /app -s /sbin/nologin bot
COPY --chown=bot:bot . .
RUN mkdir -p /app/logs /app/data && chown -R bot:bot /app/logs /app/data
USER bot

ENTRYPOINT ["python", "app.py", "live"]
CMD ["--config", "live_config_emacross.yaml"]
