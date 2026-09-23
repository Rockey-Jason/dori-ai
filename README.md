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
