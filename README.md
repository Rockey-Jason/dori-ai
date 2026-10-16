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

## 🌍 스스로 확장하는 세계 지식 수집과 실제 활용

GitHub Actions의 학습 workflow는 공개 한국어·영어 Wikipedia API를 이용해 세계 지식을 모으고, 검색 결과로 발견한 문서 제목을 바탕으로 후속 검색 주제를 새로 만든다. 기초 주제마다 핵심 개념·원리·역사·응용 검색어를 생성하고, 새로 발견한 문서마다 다시 관련 검색어 두 개를 결정적으로 생성한다. 작업 큐와 완료 기록을 저장해 다음 실행에서 이어가며, 중복 검색·중복 문서·일시적인 실패를 관리한다.

- **주간 자동 확장:** 매주 월요일 한국 시간 오전 3시 무렵 GitHub Actions가 최대 300개의 검색 주제를 처리한다. workflow_dispatch로 수동 실행도 가능하다. 큐에는 최대 8,000개의 후속 검색을 보관해 과도하게 커지는 것을 막는다.
- **학습에 포함:** 문서는 data/corpus/world_wikipedia.jsonl 및 필요할 때 생성되는 world_wikipedia_0001.jsonl 같은 20MB 이하 분할 파일에 제목·언어·원문 링크·라이선스 정보와 함께 저장된다. 기존 streaming trainer가 이 자료를 정리·중복 제거·chunking하여 학습 데이터에 포함한다.
- **실제 답변에서 활용:** dori_ai/world_knowledge.py는 먼저 공개 백과사전에서 관련 자료를 검색한다. 네트워크 검색이 실패하거나 결과가 없으면 수집된 corpus 일부를 제한된 메모리로 검색해 출처 링크가 포함된 답변 근거로 사용할 수 있다.
- **품질 보호:** 새 문서가 수집되지 않은 주간 실행에서는 불필요한 모델 학습 및 Render 배포를 건너뛴다. 새 자료가 있을 때는 기존 unit/preflight 테스트와 held-out 품질 게이트를 거친다. 후보 모델은 평가에서 개선되지 않으면 승인 checkpoint를 대체하지 않는다.
- **출처 보존:** 자료마다 원문 주소와 라이선스 정보를 남기며, corpus 전체를 메모리에 한꺼번에 올리지 않는다. 서버의 오프라인 검색은 각 파일을 스트리밍으로 읽고 검색 인덱스에 보관할 문서 수를 제한해 Render의 작은 메모리 한도에 맞춘다. 분할 파일도 모두 검색 대상에 포함된다.

### 왜 자동으로 Google 검색결과를 복사해 저장하지 않는가?

돌이 AI가 검색으로 지식을 넓히는 것은 목표지만, Google 검색결과 페이지를 무단으로 긁거나 결과 전체를 학습용 데이터베이스로 저장하는 방식은 사용하지 않는다. [Google Custom Search JSON API 안내](https://developers.google.com/custom-search/v1/overview)는 신규 고객에게 API를 제공하지 않는다고 명시하며, [Programmable Search Engine 약관](https://support.google.com/programmable-search/answer/1714300?hl=en)은 검색 결과를 크롤링·색인·지속 저장하는 행위를 제한한다. 따라서 이 저장소는 공개 API와 출처·라이선스 메타데이터가 있는 문서를 사용한다. 수집 자료를 재배포할 때는 각 원문의 라이선스와 저작자 표시 요건을 확인해야 한다.

Wikipedia 자료에는 CC BY-SA 등 라이선스 조건이 적용될 수 있다. 이 파이프라인은 주제를 넓게 확장하지만 Wikipedia 전체나 인터넷 전체를 학습한다는 뜻은 아니다. 수집량·API 가용성·GitHub 실행 자원에도 현실적인 한계가 있다.

## 📈 학습 기록

매 epoch 정보는 `runtime/training_history.jsonl`에 JSONL로 누적된다.

각 기록에는 epoch, train_loss, val_loss, gradient_norm, best_val_loss, dataset 정보와 optimizer step이 포함된다.

## 현재 모델과 checkpoint 호환

기존 `checkpoints/best.npz`는 계속 읽을 수 있다. 새 streaming checkpoint에는 optimizer 상태가 추가되지만 모델 로더는 기존 `p0`, `p1`, ... 가중치를 그대로 읽는다.

## 자동 학습

사이트의 관리자용 자동 학습 API는 기존 기능을 유지한다. 장시간의 대규모 corpus 학습은 `train_final.py`의 streaming trainer를 기본 학습 경로로 사용한다.

## 무료 데이터 기반 지속 성장 루프

이 저장소의 학습 workflow는 시간표에 따라 같은 자료를 계속 반복 학습하지 않는다. 학습 자료가 실제로 바뀌었을 때 새 성장 주기를 열고, 모델이 좋아졌는지 독립 평가한 뒤에만 새 checkpoint를 채택한다.

1. **지식 확장:** 검토한 TXT/JSON/JSONL 자료를 `data/new/`, `data/corpus/`, `data/instructions/`, `data/curriculum/`에 추가한다. 개인정보, API 키, 비공개 돌이신문 본문은 넣지 않는다.
2. **변경 감지:** 학습 자료와 tokenizer의 지문(SHA-256)을 계산한다. 동일한 지문에는 이미 진행한 회차를 이어 가며, 지문이 바뀌면 새 주기를 시작한다.
3. **가중치 이어 학습:** 새 주기의 첫 단계는 마지막으로 승인된 `checkpoints/best.npz`에서 시작하고, 그 checkpoint와 호환되는 모델 구조를 사용한다. 새로운 자료의 검증 손실은 이전 자료의 손실과 직접 비교하지 않는다.
4. **자동 품질 검사:** 문법·라우팅·검색 캐시·검색 정확도·학습 주기 테스트를 실행하고, 학습 후보는 `data/evaluation/general_qa.jsonl`의 held-out 평가를 통과해야 한다.
5. **안전한 승격 또는 복구:** 후보의 held-out 키워드 커버리지가 기존 모델보다 엄격하게 높을 때만 채택한다. 같거나 낮으면 기존 승인 checkpoint를 유지한다.

각 데이터 지문당 최대 1,200회의 학습 회차를 사용하며, 새로운 검토 자료가 추가되면 다음 주기를 다시 열 수 있다. 같은 자료를 바꾸지 않은 상태에서 반복 실행하는 것은 자동으로 막아 무료 실행 자원을 낭비하지 않는다. GitHub Actions는 `workflow_dispatch`로 직접 실행할 수도 있다.

**무료 운영의 현실적인 한계:** 데이터 업데이트와 승인된 학습 주기를 여러 번 반복할 수는 있지만, 계산 자원·GitHub 사용량·모델 크기는 무한하지 않다. 이 루프는 스스로 사실을 무제한으로 발명하는 기능이 아니라, 새 자료와 검증을 바탕으로 점진적으로 개선하고 나빠진 모델은 되돌리는 방식이다. held-out 키워드 평가는 간단한 회귀 신호이며, 전반적인 추론 능력이나 사실 정확도를 증명하지는 않는다.


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
- `scripts/evaluate_live.py`: 실행 중인 돌이 AI의 실제 답변을 질문별 루브릭으로 채점하고 응답 지연과 오류율을 기록한다. 기본값은 메모리 부담을 줄이는 8문항 카테고리 균형 표본이다.

~~~powershell
python scripts/evaluate_live.py --base-url https://dori-ai-u3kf.onrender.com --limit 8
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

## 실제 답변 품질 평가 및 안전한 모델 승격

학습 중 후보 모델은 단순히 학습 손실이나 키워드 개수만으로 승인하지 않는다. `scripts/evaluate_and_gate.py`는 기존 승인 checkpoint와 학습 후보 checkpoint를 같은 held-out 질문에 실행하고, 실제 생성된 응답을 아래 기준으로 비교한다.

- `data/evaluation/general_qa.jsonl`: 기본 지식·수학·보안 등 기존 평가 질문.
- `data/evaluation/response_quality.jsonl`: 40개의 별도 루브릭 평가 질문. 추론, 사실 지식, 한국어/영어 표현, 지시 이행, 안전한 불확실성 표현을 포함한다.
- `scripts/response_quality.py`: 질문별 핵심 기준 일치, 응답 무결성, 언어 스크립트, 반복 출력, 요구 형식 이행을 무료 규칙으로 점수화한다.
- `scripts/evaluate_and_gate.py`: 동일한 prompt와 고정 seed로 baseline/candidate 답변을 비교하고 모델 채택 여부를 결정한다.
- `scripts/evaluate_live.py`: 실행 중인 HTTP 서비스에 질문을 보내 검색·도구·라우팅을 포함한 **전체 서비스의 실제 답변**을 진단한다.

학습 후보는 전체 품질 점수가 최소 1.5%p 이상 개선되고, 핵심 정답 점수나 통과율이 유의미하게 나빠지지 않으며, 추론·안전·언어·지시 이행 카테고리에 큰 퇴보가 없고, 질문별 개선 수가 악화 수보다 많을 때만 승격한다. 조건을 통과하지 못하면 `checkpoints/best.npz`와 메타데이터를 baseline으로 복구한다.

로컬 컴퓨터에서 배포된 서비스 평가:

~~~powershell
python scripts/evaluate_live.py --base-url https://dori-ai-u3kf.onrender.com
~~~

원하는 최소 점수까지 확인하려면 `--min-quality-score 0.35`처럼 0~1 사이의 기준을 추가할 수 있다. 보고서는 기본적으로 `runtime/live_response_quality_report.json`에 저장된다. 학습 중 checkpoint 비교 보고서는 `runtime/heldout_gate_report.json`에 저장된다. 실패 요청이 한 번이라도 발생하면 기본적으로 추가 요청을 중단하고, 실패 응답을 오답으로 섞지 않은 채 성공 응답만 따로 채점한다. `--limit 0`은 전체 60문항을 뜻하지만, 512 MiB 무료 서버에 연속 요청을 보내면 메모리 한도를 넘길 수 있으므로 실서비스 진단에서는 권장하지 않는다. 전체 60문항의 모델 간 비교는 서버 요청 없이 수행하는 checkpoint 평가 게이트가 담당한다.

이 평가는 외부 AI 심사자나 유료 API를 사용하지 않는 재현 가능한 회귀 시험이다. 루브릭이 명시한 사실과 형식은 자동으로 검사할 수 있지만, 개방형 의미 이해와 문법의 모든 미묘함을 완벽하게 판정하지는 못한다. 따라서 승격은 이 고정 시험에서의 검증된 개선을 의미하며, 모든 영역에서 지능이 향상되었다는 증명은 아니다. 평가 데이터는 학습 데이터에 섞지 않고 계속 held-out 상태로 유지한다.
