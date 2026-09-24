# 🐶 Dori AI 3.0 — From-Scratch + Evidence-First Architecture

돌이 AI 3.0은 **외부 생성형 AI 모델/API나 사전학습 가중치를 사용하지 않고** 직접 구현한 NumPy Transformer를 중심으로 구성한 프로젝트다.

> 현실적인 한계: 이 프로젝트의 작은 Transformer는 ChatGPT 같은 프런티어 모델과 동일한 규모/성능이 아니다. 수백만~수십억 개 이상의 파라미터와 거대한 학습 데이터/컴퓨팅 자원이 필요한 영역이기 때문이다. 대신 돌이 AI는 **실시간 돌이사이트 데이터 + 로컬 지식 검색 + 안전한 계산기 + 웹 검색 + 대화 메모리 + 자체 Transformer**를 결합해 작은 모델의 약점을 보완한다.

## 핵심 기능

- 🧠 직접 구현한 causal Transformer
- 🔤 lossless byte-level BPE tokenizer
- 📚 984+개의 로컬 Q&A/지식 검색 항목과 corpus
- 🌐 Supabase에서 돌이신문/퀴즈/돌돌증권/사용자 정보를 실시간 조회
- 🔐 Supabase access token을 서버에서 검증하여 사용자별 열람 권한 적용
- 📰 사용자가 읽을 수 있는 돌이신문 번호만 조회
- 📈 돌돌증권의 현재 종목/가격/변동/위험도 조회
- 🔎 최신성 질문에 선택적 웹 검색
- 🧮 안전한 수학 계산기 + 한국어 산술 표현 지원
- 💬 사용자별 대화 기록과 장기 메모리
- ⚡ fast / deep 응답 모드
- 🧠 관리자 전용 자동 학습 버튼 및 `/training/*` API
- ♻️ 자동 학습 전 기존 checkpoint 백업
- 📡 HTTP + SSE 스트리밍
- 🛡️ 특수 토큰/반복/비정상 생성 필터
- 🔄 지식 reload API

## 데이터와 학습

현재 제공되는 체크포인트는 업로드된 학습 결과 중 validation loss가 가장 낮았던 **epoch 350** 모델이다.

- vocab: 768
- context: 128
- dim: 64
- heads: 4
- layers: 3
- ff: 256
- best validation loss: 약 3.54495
- corpus: `data/corpus/v25_training.txt`

기본 재학습:

```powershell
python train_final.py --data data/corpus/v25_training.txt --epochs 500 --seq-len 128 --dim 64 --heads 4 --layers 3 --ff-dim 256 --lr 2e-4 --grad-clip 1.0
```

## 자동 학습

사이트의 **🧠 자동 학습 시작** 버튼은 관리자(`is_admin=true` 및 `user_level >= 10`)에게만 표시된다.

학습 모드는 다음 자료를 모은다.

1. `data/corpus/*.txt`
2. `data/*.jsonl`
3. `data/generated/*.jsonl`
4. 현재 Supabase의 돌이신문
5. 현재 Supabase의 돌돌증권
6. 자동 생성 수학 데이터

그 뒤 기존 Transformer에 continued language-model training을 수행하고 서버 모델을 hot-reload한다.

### 중요한 점

자동 학습은 **스스로 무한히 새로운 지식을 발명하는 AGI 기능이 아니다.** 제공된 데이터와 알고리즘으로 기존 모델의 가중치를 추가 학습하는 기능이다. 장시간 학습은 Render Free 같은 수면형 서비스보다 지속 실행 가능한 컴퓨팅 환경이 적합하다.

## 서버 실행

```powershell
python server.py
```

필수 환경변수:

```text
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
```

선택 환경변수:

```text
DORI_WEB_SEARCH=1
DORI_TRAIN_STEPS=600
DORI_MATH_EXAMPLES=12000
DORI_TRAIN_MICROBATCH=4
DORI_TRAIN_LR=1.5e-4
```

## API

- `GET /health`
- `POST /chat`
- `GET /training/status` — 관리자
- `POST /training/start` — 관리자
- `POST /training/stop` — 관리자
- `POST /knowledge/reload` — 관리자

## 원칙

- OpenAI / Gemini / Claude / Ollama / Gemma / Llama / Qwen / Hugging Face pretrained weights를 핵심 모델로 사용하지 않는다.
- 실시간 돌이사이트 데이터는 가능한 경우 **학습된 모델보다 우선**하여 읽는다.
- 최신 정보는 선택적으로 웹 검색 결과를 사용한다.
- 확인할 수 없는 정보는 작은 Transformer가 억지로 지어내지 않도록 품질 필터를 적용한다.
- 브라우저가 보낸 `user_id`만으로 권한을 결정하지 않고 서버에서 access token을 검증한다.
