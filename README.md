# 🐶 Dori AI 3.1 — World Knowledge + Evidence-First Architecture

돌이 AI는 직접 구현한 NumPy Transformer, 다국어 공개 지식 수집·검색, 대화 문맥 처리, 근거 기반 답변을 결합한다. 외부 AI API는 사용하지 않는다. 자체 서버에서 실행하는 사전학습 모델은 선택적으로 private/local endpoint로 연결할 수 있으며, 실제 모델 가중치와 실행 자원은 운영자가 직접 제공한다. 작은 자체 제작 Transformer만으로 세상의 모든 지식을 학습했다고 주장하지 않는다.

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
11. 🛑 validation 성능이 일정 epoch 동안 개선되지 않으면 조기 종료해 과적합을 줄임

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
python train_final.py --epochs 100 --patience 10 --min-delta 0.001 --seq-len 128 --batch-size 8 --dim 64 --heads 4 --layers 3 --ff-dim 256 --lr 2e-4 --grad-clip 1.0
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

## 🌍 다양한 세계 지식 수집과 실제 활용

GitHub Actions의 학습 workflow는 이제 학습을 시작하기 전에 `scripts/collect_world_knowledge.py`를 실행한다. 이 수집기는 한국어와 영어 Wikipedia의 공개 API에서 수학, 자연과학, 컴퓨터 과학, 역사, 지리, 경제, 법, 철학, 예술, 언어, 환경, 공학, 건강 일반지식 등 여러 주제의 요약을 가져온다.

- 가져온 문서는 `data/corpus/world_wikipedia.jsonl`에 제목·언어·원문 링크·라이선스 정보를 붙여 저장한다.
- 수집된 텍스트는 기존 `CorpusBuilder`에 의해 정리·중복 제거·chunking된 뒤 `train_final.py`의 학습 데이터에 포함되어 Transformer 가중치 업데이트에 실제로 사용된다.
- `dori_ai/world_knowledge.py`는 사용자 질문에 따라 Wikipedia에서 관련 문서를 검색하고, 결과를 캐시에 저장한다.
- 검색 근거는 로컬/자체 호스팅 모델이 설정된 경우에만 모델 입력으로 전달된다. 모델이 없으면 출처 링크가 포함된 백과사전 요약을 직접 반환해 무작위 Transformer 출력을 피한다.
- 후속 질문은 직전 사용자 질문과 연결해 검색하도록 기본 처리를 추가했다.

Wikipedia 텍스트의 재사용에는 CC BY-SA 등 라이선스 조건이 적용될 수 있다. 출처 메타데이터를 보존하고, 재배포 전에 관련 조건을 확인한다. 이 수집기는 제한된 주제 목록을 대상으로 하므로 Wikipedia 전체를 학습하는 것은 아니다.

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

## 자체 서버의 사전학습 모델 (선택 기능)

외부 AI API는 사용하지 않는다. 운영자가 직접 관리하는 llama.cpp 호환 추론 서버만 연결할 수 있다. 설정된 호스트가 실제로 모델을 실행하고 있어야 하며, 연결하지 않으면 자체 Transformer와 검색/규칙 기반 경로가 사용된다.

Windows 설치·실행 절차, 모델 크기 및 보안 연결 방식은 [`local-model/README.md`](local-model/README.md)를 참고한다. 추천 시작점은 Qwen3-4B-GGUF Q4_K_M이지만 가중치만 약 2.6GB이고 실행 메모리도 추가로 필요하다. 현재 Render Free 512MB 인스턴스 안에서 이 모델을 실행하면 메모리 초과가 발생할 가능성이 높으므로 별도의 본인 소유 호스트가 필요하다.

환경 변수:

~~~text
DORI_LOCAL_LLM_URL=https://your-own-model-host.example.net
DORI_LOCAL_LLM_MODEL=Qwen3-4B-GGUF:Q4_K_M
DORI_LOCAL_LLM_TOKEN=<your-own-long-random-secret>
DORI_LOCAL_LLM_TIMEOUT=25
DORI_LOCAL_LLM_MAX_TOKENS=600
~~~

공개 AI 추론 제공자 주소는 코드에서 거부한다. 공개 HTTPS 호스트는 명시적인 토큰이 설정된 경우에만 허용한다. 토큰을 GitHub에 커밋하거나 채팅에 붙여 넣지 않는다. Render에서 localhost는 사용자 PC의 localhost가 아니므로, 자체 호스트 연결에는 보안이 설정된 사설 네트워크 또는 인증된 HTTPS 프록시가 필요하다.

## 검증된 학습 자료 및 독립 평가

- `data/curriculum/qa_following/verified_general_qa.jsonl`: 한국어 수학·과학·컴퓨터 과학·AI·보안·돌이 세계관을 포함한 검토용 시드 Q&A 54개.
- `data/evaluation/general_qa.jsonl`: 학습 자료와 분리된 20개 평가 질문. 평가 질문은 학습 데이터로 복사하지 않는다.
- `scripts/collect_world_knowledge.py`: 공개 백과사전에서 다국어 지식을 수집하는 기존 파이프라인.
- `scripts/evaluate_live.py`: 실행 중인 돌이 AI에 평가 질문을 보내 키워드 커버리지와 응답 지연을 측정한다.

~~~powershell
python scripts/evaluate_live.py --base-url https://dori-ai-u3kf.onrender.com
~~~

키워드 커버리지는 빠른 회귀 신호일 뿐 이해력이나 사실 정확도를 완전히 측정하지 않는다. 실패 답변을 검토하고, 별도 평가 질문으로 다시 시험해야 한다.

## 돌이신문 및 사이트 검색 권한

- 돌이신문 본문 검색은 로그인한 사용자의 `read_dori_news` 한도 이하 기사만 조회한다.
- 열람 한도보다 높은 호수는 검색 결과·본문·퀴즈를 반환하지 않는다.
- 돌이신문 본문은 정적 학습 corpus에 복사하지 않는다. 권한이 바뀔 수 있는 콘텐츠는 질문 시점에 인증된 사용자 기준으로 조회한다.
- 돌이사이트 관련 질문은 자체 사이트 자료와 로컬 지식만 사용하고, 확인할 수 없는 내용을 생성형 추측으로 채우지 않는다.
- 일반 지식은 공개 백과사전 검색과 로컬 자료를 활용한다. 검색 실패나 출처 부족 시 불확실성을 알려야 한다.

## 원칙

- OpenAI / Gemini / Claude 등 외부 AI API는 사용하지 않는다. 자체 호스팅 사전학습 모델은 선택 기능이며 공개 endpoint를 허용하지 않는다.
- 실시간 돌이사이트 데이터는 가능한 경우 학습된 모델보다 우선하여 읽는다.
- 최신 정보는 선택적으로 웹 검색 결과를 사용한다.
- 작은 Transformer가 확인할 수 없는 정보를 억지로 지어내지 않도록 품질 필터를 적용한다.
- 브라우저가 보낸 `user_id`만으로 권한을 결정하지 않고 서버에서 access token을 검증한다.
