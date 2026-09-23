# 🐶 Dori AI — Final Training Edition

This is a from-scratch educational Dori AI stack.

## Core
- Hand-written NumPy Transformer
- Lossless byte-level BPE tokenizer
- Hand-written Adam optimizer/autograd
- Train/validation split
- Atomic-ish latest/best checkpoint pair with metadata and tokenizer SHA-256
- Persistent local memory
- Dialogue context
- Local Dori knowledge retrieval for fast, stable factual replies
- Transformer fallback for unseen prompts
- Generation quality gate to avoid obvious repetition/garbage

No OpenAI/Gemini/Claude API, Ollama, Gemma, Llama, Qwen, Hugging Face pretrained model, or external pretrained weights are used.

## Run immediately
The ZIP contains a checkpoint trained by this project on the bundled Dori corpus.

```bat
chat_final.bat
```

## Verify
```bat
test_final.bat
```

## Rebuild and train from scratch
```bat
build_and_train_final.bat
```

The bundled model is intentionally small. Its local Dori knowledge is accurate only within the supplied corpus; unknown/current-world questions are not magically known without a data source. The system therefore prefers a verified local answer or says it does not know rather than inventing facts.


## Dori Knowledge Expansion
The project now includes a curated, explicit-status Dori knowledge base at `data/dori_knowledge.jsonl` and a human-readable corpus at `data/corpus/dori_knowledge_corpus.txt`. It expands paraphrased Q&A without inventing new lore. Retrieval loads both the knowledge base and instruction data.

## Dori AI v2.3 expansion

v2.3 adds a much larger curated reference layer and an optional public-web source finder.

- data/dori_knowledge_expanded.jsonl: expanded Dori/site/world knowledge
- data/dori_knowledge_v23.jsonl: additional stable math, science, history, geography, programming, English, chess and reasoning references
- data/corpus/v23_training.txt: generated 2,868-record supervised corpus
- dori_ai/retrieval.py: stronger exact/containment/ngram/token/topic retrieval
- dori_ai/web_search.py: optional DuckDuckGo HTML source finder using only Python stdlib; no AI API
- dori_ai/response_engine.py: local knowledge -> web source search -> from-scratch Transformer fallback
- build_v23_dataset.py: rebuilds the corpus from the curated JSONL sources
- train_final.py: v2.3 training defaults and optional tokenizer retraining

### Train v2.3

python build_v23_dataset.py
python train_final.py --retrain-tokenizer --epochs 500 --seq-len 128 --dim 64 --heads 4 --layers 3 --ff-dim 256

The current repository checkpoint remains usable while v2.3 is being trained. The new retrieval and web-search layers can operate independently of a newly trained checkpoint.

Web search is a source-finding layer, not an external language model. Search can fail because of network restrictions; in that case Dori AI falls back to its local knowledge and its own Transformer.


## Dori AI v2.3 site integration

- Multilingual lightweight detection/normalization for Korean, English, Japanese, Chinese, Spanish, French, German, Russian, Portuguese and Italian.
- Read-only Supabase bridge for current Dori newspaper and Doldol Securities data.
- Newspaper access is enforced server-side using `users.read_dori_news`. A requested issue above that user's readable number is never returned.
- A frontend may send `user_id` with `POST /chat` so the server can enforce the newspaper limit.
- Newspaper quiz questions/options can be shown without exposing the stored answer key.
- Stock answers are read from the current `dori_stocks` table, so prices are not baked into the model.

Required Render environment variables for site integration:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `DORI_WEB_SEARCH=1` (optional; enabled by default)
