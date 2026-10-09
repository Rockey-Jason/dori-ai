# 🐶 Dori AI 3.0 — From-Scratch + Evidence-First Architecture

돌이 AI 3.0은 외부 생성형 AI 모델/API나 사전학습 가중치를 사용하지 않고 직접 구현한 NumPy Transformer를 중심으로 구성한 프로젝트다.

## 🚀 대용량 학습 파이프라인

`train_final.py`는 전체 corpus를 한 번에 토큰 배열로 RAM에 올리는 대신 디스크 기반 streaming dataset을 사용한다.

### 자동 처리되는 10단계

1. 📂 TXT / JSON / JSONL 파일 재귀 수집
2. 🧹 공백/제어문자 정리, 반복 문자·비정상 레코드 제거, 중복 제거
3. ✂️ 문단/문장 경계를 우선한 chunking
4. 🔤 custom byte-level BPE tokenizer 적용
5. 💾 디스크의 JSONL chunk를 batch 단위로 읽어 학습
6. 🔄 모델 + Adam optimizer 상태를 checkpoint에서 이어 학습
7. 🧠 기존 + 신규 데이터 폴더를 함께 수집
8. 📊 결정적 hash 기반 train/validation 자동 분리
9. 📈 epoch별 train/validation loss를 기록
10. 💾 validation loss가 가장 낮은 모델을 best checkpoint로 자동 저장

### 기본 데이터 위치

~~~text
data/
├── corpus/       # 기존 학습 데이터
├── new/          # 새 TXT/JSON/JSONL
├── generated/    # 생성/가공 데이터
└── tokenizer.json

data/streaming/
├── train.jsonl
├── val.jsonl
└── manifest.json
~~~

`data/new/`에 새 TXT/JSON/JSONL을 넣으면 기존 corpus와 함께 다음 학습에 포함된다.

### 기본 학습

~~~powershell
python train_final.py --epochs 10 --seq-len 128 --batch-size 8 --dim 64 --heads 4 --layers 3 --ff-dim 256 --lr 2e-4 --grad-clip 1.0
~~~

### 외부 파일/폴더 추가

~~~powershell
python train_final.py --data "C:\mydata\science" --data "C:\mydata\qa.jsonl" --epochs 10
~~~

### BPE tokenizer 재학습

~~~powershell
python train_final.py --retrain-tokenizer --fresh --epochs 10
~~~

새 tokenizer는 token ID 의미가 달라질 수 있으므로 안전을 위해 새 모델로 시작한다.

### checkpoint에서 이어 학습

~~~powershell
python train_final.py --epochs 10
~~~

`checkpoints/latest.npz`가 있으면 자동으로 이어 학습한다. 새 checkpoint에는 Transformer 가중치와 Adam의 `m`, `v`, `t` 상태가 함께 저장된다.

완전히 새로 시작:

~~~powershell
python train_final.py --fresh --epochs 10
~~~

### 기존 streaming cache 재사용

~~~powershell
python train_final.py --no-build-cache --epochs 10
~~~

새 파일을 추가했거나 원본을 수정했다면 기본 실행으로 cache를 다시 만든다.

## 📈 학습 기록

매 epoch 정보는 `runtime/training_history.jsonl`에 JSONL로 누적된다.

각 기록에는 epoch, train_loss, val_loss, gradient_norm, best_val_loss, dataset 정보와 optimizer step이 포함된다.

## 현재 모델과 checkpoint 호환

기존 `checkpoints/best.npz`는 계속 읽을 수 있다. 새 streaming checkpoint에는 optimizer 상태가 추가되지만 모델 로더는 기존 `p0`, `p1`, ... 가중치를 그대로 읽는다.

## 자동 학습

사이트의 관리자용 자동 학습 API는 기존 기능을 유지한다. 장시간의 대규모 corpus 학습은 `train_final.py`의 streaming trainer를 기본 학습 경로로 사용한다.

## 서버 실행

~~~powershell
python server.py
~~~

필수 환경변수:

~~~text
SUPABASE_URL=...
SUPABASE_ANON_KEY=...
~~~

선택 환경변수:

~~~text
DORI_WEB_SEARCH=1
DORI_TRAIN_STEPS=600
DORI_MATH_EXAMPLES=12000
DORI_TRAIN_MICROBATCH=4
DORI_TRAIN_LR=1.5e-4
~~~

## 폭넓은 대화형 AI 제공자 (선택 기능)

기본 from-scratch NumPy Transformer는 학습 규모와 문맥 길이에 한계가 있어, 다양한 주제의 지식과 자연스러운 대화를 단독으로 보장할 수 없다. 이를 보완하기 위해 OpenAI 호환 Chat Completions API를 선택적으로 연결할 수 있다.

Render의 Dori AI Web Service 환경 변수에 다음 값을 설정하면 일반 질문과 문맥을 잇는 후속 질문을 외부 사전학습 모델로 처리한다. API 키는 서버 환경 변수에만 저장하며 브라우저로 보내지 않는다.

~~~text
DORI_LLM_API_KEY=your_api_key
DORI_LLM_MODEL=gpt-4.1-mini
DORI_LLM_BASE_URL=https://api.openai.com/v1
DORI_LLM_TIMEOUT=35
DORI_LLM_MAX_TOKENS=700
~~~

기존 `OPENAI_API_KEY` 환경 변수가 이미 있으면 `DORI_LLM_API_KEY` 대신 사용할 수도 있다. 다른 OpenAI 호환 제공자를 사용할 경우 해당 제공자의 기본 URL과 모델 이름을 지정한다. 제공자가 설정되지 않거나 요청에 실패하면 기존 로컬 경로로 대체된다. `/health`는 키를 노출하지 않고 제공자 활성 여부와 모델 이름만 보여준다.

최신 정보가 필요한 질문은 웹 검색 결과를 모델에 참고 자료로 전달하도록 구성했다. 검색 결과는 신뢰할 수 없는 자료로 취급하며, 검색 자체가 실패할 수 있으므로 중요한 사실은 출처를 확인해야 한다.

## 원칙

- OpenAI / Gemini / Claude / Ollama / Gemma / Llama / Qwen / Hugging Face pretrained weights를 핵심 모델로 사용하지 않는다.
- 실시간 돌이사이트 데이터는 가능한 경우 학습된 모델보다 우선하여 읽는다.
- 최신 정보는 선택적으로 웹 검색 결과를 사용한다.
- 작은 Transformer가 확인할 수 없는 정보를 억지로 지어내지 않도록 품질 필터를 적용한다.
- 브라우저가 보낸 `user_id`만으로 권한을 결정하지 않고 서버에서 access token을 검증한다.
