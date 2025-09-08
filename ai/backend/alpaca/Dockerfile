#!/usr/bin/env bash
# backend/alpaca/start_portfolio_service.sh

FROM python:3.13-slim AS base

WORKDIR /app/backend/alpaca
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
EXPOSE 8001


FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY req.txt .
RUN pip install --no-cache-dir -r req.txt


FROM base AS prod
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . .
RUN chmod +x start_portfolio_service.sh
CMD ["./start_portfolio_service.sh"]


FROM base AS dev
COPY --from=builder /opt/venv /opt/venv
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/opt/venv/bin:$PATH"
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["./start_portfolio_service.sh"]