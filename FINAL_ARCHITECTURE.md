# Dori AI Final Architecture

This package is the strongest practical **from-scratch** Dori AI build in this repository. It intentionally does not use OpenAI/Gemini/Claude/Ollama/Hugging Face pretrained weights or another company's language model.

## What is included

- NumPy Transformer trained from scratch.
- Byte-level BPE tokenizer with lossless UTF-8 round trips.
- Local Dori knowledge retrieval with paraphrase-aware matching.
- Live Supabase access for Dori newspaper, user-readable newspaper limits, Doldol Securities, and user profile data.
- Optional live web search for time-sensitive questions.
- Per-user conversation memory.
- Deterministic math engine instead of asking the tiny language model to calculate.
- Admin-only automatic learning mode.
- Automatic collection of local corpus + generated datasets + current Dori newspaper + current stock information.
- Atomic checkpoint replacement with a pre-learning backup.
- Runtime status endpoint and frontend admin learning button.
- Fast/deep routing. Deep mode gives the from-scratch model a longer generation budget; it is not a second external model.
- CORS and Supabase access-token verification.

## Important engineering limitation

This is **not literally a ChatGPT-scale foundation model**. A 64-dimensional, 3-layer NumPy Transformer with roughly 1M characters of training text cannot reproduce the capabilities of a frontier model trained on enormous datasets with massive compute. Adding arbitrary data cannot remove that limitation.

The package therefore uses an evidence-first architecture: live site data and deterministic tools are preferred for factual answers; local knowledge is preferred next; web search handles freshness; the small Transformer is the open-ended fallback. This makes the system much more reliable than forcing the small model to memorize every fact.

## Automatic learning

The button is exposed only when the current Supabase access token belongs to an administrator (`is_admin=true` and `user_level >= 10`). The server never trusts a browser-supplied user ID for authorization.

Learning is continued language-model training, not magical self-improvement. It can be stopped and the previous checkpoint is retained as `best.prelearning.npz`.

For true always-on long training, use a persistent compute service rather than a sleeping free web service.
