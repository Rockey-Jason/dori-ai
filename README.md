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
10. 💾 validation loss가 가장 낮은 모델을 best checkpoint로 자동 저장\n11. 🛑 validation 성능이 일정 epoch 동안 개선되지 않으면 조기 종료해 과적합을 줄임

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

외부 AI API 키를 사용하지 않는다. 필요하면 운영자가 관리하는 **자체 호스팅 llama.cpp 호환 서버**를 \`DORI_LOCAL_LLM_URL\`로 연결할 수 있다. URL은 localhost, 사설 IP 또는 내부 호스트 이름만 허용하며, 공개 AI 서비스 주소는 코드에서 거부한다.

~~~text
DORI_LOCAL_LLM_URL=http://127.0.0.1:8080
DORI_LOCAL_LLM_MODEL=dori-local-model
DORI_LOCAL_LLM_TIMEOUT=20
DORI_LOCAL_LLM_MAX_TOKENS=600
~~~

예시 주소는 로컬 서버가 실제로 실행 중일 때만 작동한다. Render의 현재 작은 메모리 인스턴스에서 큰 사전학습 모델을 함께 실행한다고 가정하지 않는다. 모델을 실행할 별도 자체 호스트가 필요할 수 있으며, 연결하지 않으면 Dori AI는 자체 Transformer와 검색/규칙 기반 경로를 사용한다. \`llama.cpp\`는 양자화 모델을 CPU에서 실행할 수 있지만, 모델 크기와 속도는 실제 자원에 따라 달라진다.

## 돌이신문 및 사이트 검색 권한

- 돌이신문의 본문 검색은 로그인한 사용자의 \`read_dori_news\` 한도 이하의 기사만 DB에서 가져온다.
- 열람 한도보다 높은 호수는 검색 결과·본문·퀴즈를 반환하지 않는다.
- 돌이신문 본문은 정적 학습 corpus에 복사하지 않는다. 권한이 달라질 수 있는 콘텐츠는 질문 시점에 인증된 사용자 기준으로 조회한다.
- 돌이사이트 관련 질문은 자체 사이트 자료와 로컬 지식만 사용하고, 확인할 수 없는 내용을 일반 웹 검색이나 생성형 추측으로 채우지 않는다.
- 일반 지식은 공개 백과사전 검색과 로컬 자료를 활용한다. 인터넷 검색이 꺼져 있거나 실패할 수 있으므로 출처가 확인되지 않은 사실은 불확실하다고 알린다.

## 원칙

- OpenAI / Gemini / Claude 등 외부 AI API는 사용하지 않는다. 자체 호스팅 사전학습 모델은 선택 기능이며 공개 endpoint를 허용하지 않는다.
- 실시간 돌이사이트 데이터는 가능한 경우 학습된 모델보다 우선하여 읽는다.
- 최신 정보는 선택적으로 웹 검색 결과를 사용한다.
- 작은 Transformer가 확인할 수 없는 정보를 억지로 지어내지 않도록 품질 필터를 적용한다.
- 브라우저가 보낸 `user_id`만으로 권한을 결정하지 않고 서버에서 access token을 검증한다.
