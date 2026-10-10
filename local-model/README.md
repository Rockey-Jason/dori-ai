# Dori AI self-hosted pretrained model

This is a project-owned inference service, not a third-party AI API. It is optional until a machine with sufficient memory is available. Do not try to load a multi-billion-parameter model inside the current 512 MB Render Free web service; it is likely to exceed the memory limit and restart.

## Suggested model

Start with the official [Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) Q4_K_M quantization for a useful general-purpose baseline. The published file is around 2.6 GB before runtime memory and context cache, so plan for several GB of free RAM. The model card lists Apache-2.0; review its current license and terms before redistribution. Smaller GGUF models may fit smaller machines, with a possible quality tradeoff.

## Windows quick start

1. Install llama.cpp from a terminal:

   ```powershell
   winget install llama.cpp
   ```

2. Start the model server:

   ```powershell
   llama-server -hf Qwen/Qwen3-4B-GGUF:Q4_K_M --host 127.0.0.1 --port 8080 --ctx-size 4096 --n-gpu-layers 0
   ```

   The first start downloads model weights from Hugging Face. Later starts can reuse the cache. Keep the server bound to loopback while testing.

3. Test the model locally in another PowerShell window:

   ```powershell
   $body = '{"model":"Qwen/Qwen3-4B-GGUF:Q4_K_M","messages":[{"role":"user","content":"대한민국의 수도는 어디야?"}]}'
   Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType 'application/json' -Body $body
   ```

## Connecting the online Dori AI service

A Render process cannot reach 127.0.0.1 on your own PC. For the online service to use your own model, the model must run on a machine you control that Render can reach through private networking or a secured HTTPS endpoint. Do not expose port 8080 directly to the public internet. Use an authenticated reverse proxy or private networking and configure these Render environment variables only after the host is secured:

- `DORI_LOCAL_LLM_URL=https://your-own-model-host.example.net` (base URL, without `/v1/chat/completions`)
- `DORI_LOCAL_LLM_MODEL=Qwen3-4B-GGUF:Q4_K_M`
- `DORI_LOCAL_LLM_TOKEN` (a long random secret configured in your own proxy)

The adapter rejects known third-party AI inference API hosts. The bearer token is sent only to the explicitly configured self-hosted endpoint. Never commit the token or paste it into chat. This connection is not active until the model server is running and environment variables are configured.

## Evaluate instead of guessing

After the model is connected and the Render deployment is live:

```powershell
python scripts/evaluate_live.py --base-url https://dori-ai-u3kf.onrender.com
```

This tests a held-out 20-question set and reports keyword coverage and latency. It is only a regression signal, not proof of general intelligence. Review failed answers and add independent test cases before training on them; do not move evaluation questions into training data.
