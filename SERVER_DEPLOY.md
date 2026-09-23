# Dori AI 24/7 deployment package

This package turns the current Dori AI model into a server process that survives your PC being turned off because the model runs on a cloud VM.

Recommended low-cost/free-tier target: Google Cloud Compute Engine e2-micro. The Google Free Tier currently includes one non-preemptible e2-micro per month in supported US regions, 30 GB standard persistent disk, and 1 GB/month outbound transfer for eligible users. A billing account is required. Limits can change; verify the current official terms before deployment.

The package also includes a systemd service, nginx reverse proxy, health endpoint, Dockerfile, and update script.
