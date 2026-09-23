#!/usr/bin/env bash
set -euo pipefail
APP=/opt/dori-ai
SRC=${1:-$(pwd)}
apt-get update
apt-get install -y python3 python3-venv nginx
id -u dori >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin dori
mkdir -p "$APP"
cp -a "$SRC"/. "$APP"/
chown -R dori:dori "$APP"
python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install --upgrade pip
"$APP/.venv/bin/pip" install -r "$APP/requirements.txt"
cp "$APP/deploy/dori-ai.service" /etc/systemd/system/dori-ai.service
cp "$APP/deploy/dori-nginx.conf" /etc/nginx/sites-available/dori-ai
ln -sf /etc/nginx/sites-available/dori-ai /etc/nginx/sites-enabled/dori-ai
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl daemon-reload
systemctl enable --now dori-ai
systemctl enable --now nginx
systemctl --no-pager --full status dori-ai || true
curl -fsS http://127.0.0.1:8000/health
