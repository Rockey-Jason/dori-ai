# Dori AI 24/7 server deployment

Recommended first deployment target: Google Cloud Compute Engine Free Tier eligible e2-micro in a supported US region. Current Google docs list one non-preemptible e2-micro/month, 30 GB standard persistent disk, and 1 GB/month outbound transfer under the Free Tier; a billing account is required. See Google's current Free Tier docs before creating resources.

## 1. Create VM
Ubuntu LTS, e2-micro, supported Free Tier US region. Reserve a static external IP only if you understand its billing rules; otherwise use the VM external IP.

## 2. Open firewall
Allow TCP 80. For setup/SSH, use TCP 22 from your own IP when possible.

## 3. Upload this project
From your PC, copy the extracted `dori-ai-final` folder to `/opt/dori-ai` (or upload the ZIP and extract it). Keep checkpoints and `data/` with the code.

## 4. Run setup as root
`sudo deploy/setup_ubuntu.sh`

## 5. Verify
`curl http://127.0.0.1:8000/health`
Then visit `http://VM_EXTERNAL_IP/`.

## 6. HTTPS/custom domain
Point an A/AAAA record to the VM. Install Caddy or configure nginx + Let's Encrypt after DNS resolves. Do not expose port 8000 publicly; only nginx should receive public HTTP/HTTPS.

## 7. Logs
`sudo journalctl -u dori-ai -f`

## 8. Restart/update
`sudo systemctl restart dori-ai`

The model is self-hosted and does not call an external AI model API.
