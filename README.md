# 🐶 Dori AI — multilingual + Dori site integration

This package is the current Dori AI from-scratch project bundle. The core model is a hand-written NumPy Transformer; no OpenAI/Gemini/Claude API, Ollama, Gemma, Llama, Qwen, Hugging Face pretrained model, or external pretrained weights are used.

## Included
- Hand-written NumPy Transformer + autograd/Adam
- Byte-level BPE tokenizer
- Local Dori knowledge retrieval and dialogue memory
- Existing trained checkpoint
- Multilingual detection/normalization for Korean, English, Japanese, Chinese, Spanish, French, German, Russian, Portuguese and Italian
- Read-only Supabase bridge for current Dori newspaper and Doldol Securities data
- Per-user `read_dori_news` newspaper access limit
- Newspaper quiz display without exposing the answer key
- Optional public-web source finder
- Render-ready `server.py`
- Connected Dori AI frontend in `frontend/dori-ai-connected.html`

## Newspaper access rule
The server accepts `user_id` and checks `users.read_dori_news` before returning a requested `rockey_news` row. If the requested issue is higher than the user's readable number, the article is refused.

## Supabase / Render settings
Set on Render:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `DORI_WEB_SEARCH=1` (optional)

The real Doldol site should pass the currently authenticated user's Supabase ID to `/chat`. The bundled frontend supports `window.DORI_USER_ID` or `localStorage.dori_user_id` for compatibility.

See `INTEGRATION_NOTES.md` for the production security note about JWT verification.

## Local run
```bat
chat_final.bat
```

Server:
```bat
python server.py
```

Validation:
```bat
test_final.bat
```

## Retraining
The included checkpoint is the already-trained model from the previous project stage. Retraining is separate from the live site-data bridge: the bridge reads current newspaper/stock data directly from Supabase instead of baking changing prices or issue contents into model weights.
