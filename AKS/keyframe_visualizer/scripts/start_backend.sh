#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../backend"
exec python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
