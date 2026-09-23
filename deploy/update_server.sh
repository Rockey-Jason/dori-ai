#!/usr/bin/env bash
set -euo pipefail
APP=/opt/dori-ai
SRC=${1:-$(pwd)}
rsync -a --delete --exclude '.venv' --exclude '__pycache__' "$SRC/" "$APP/"
chown -R dori:dori "$APP"
"$APP/.venv/bin/pip" install -r "$APP/requirements.txt"
systemctl restart dori-ai
curl -fsS http://127.0.0.1:8000/health
