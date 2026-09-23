# 🐶 Dori AI 2.5

돌이 AI는 **외부 생성형 AI 모델/API 없이 직접 구현한 NumPy Transformer**를 중심으로 동작한다.

## 핵심 기능

- 🧠 직접 구현한 Transformer + causal self-attention
- 📚 BPE 토크나이저 + 대규모 로컬 지식 검색
- 🔥 관리자 전용 자동 딥러닝 학습 모드
- 🛡️ 학습 전 체크포인트 자동 백업 + 원자적 교체
- 🧮 안전한 수학 계산기: 사칙연산, %, 거듭제곱, 함수, 1차방정식
- 🌍 한국어/영어/일본어/중국어/스페인어/프랑스어/독일어/포르투갈어/이탈리아어/러시아어 인식 및 응답 템플릿
- 🌐 돌이사이트 Supabase 연동: 로그인 사용자 정보, 돌이신문 열람 권한, 돌이신문, 퀴즈, 돌돌증권
- 🔎 최신/현재성 질문에 대한 선택적 웹 검색 어댑터
- 💬 사용자별 대화 기록과 장기 메모리
- 📡 HTTP + SSE 스트리밍 서버
- 🔐 Supabase access token 검증 후 사용자별 권한 처리
- 🧹 반복/깨진/특수 토큰 출력 품질 필터

## 학습

기본 학습은 `train_final.py`로 수행한다. 서버의 `/training/start`는 관리자만 호출할 수 있으며, 로컬 corpus + 지식 JSONL + 돌이사이트 공개 데이터 + 자동 생성 수학 데이터로 **continued language-model training**을 수행한다.

> 자동 학습은 사람이 정한 데이터와 규칙을 이용해 기존 Transformer 가중치를 추가 학습하는 방식이다. 스스로 무한히 새로운 지식을 만들어내는 AGI가 아니다.

수학 데이터 생성:

```bash
python -m dori_ai.data_tools
```

기본 학습 예시:

```bash
python train_final.py --retrain-tokenizer --epochs 500 --seq-len 128 --dim 64 --heads 4 --layers 3 --ff-dim 256
```

## 서버

```bash
python server.py
```

환경변수:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `DORI_WEB_SEARCH=1`
- `DORI_TRAIN_STEPS=600`
- `DORI_MATH_EXAMPLES=12000`

### API

- `GET /health`
- `POST /chat`
- `GET /training/status` (관리자)
- `POST /training/start` (관리자)
- `POST /training/stop` (관리자)

`POST /chat` 예시:

```json
{"message":"2의 10제곱은?", "stream":true}
```

## 중요한 원칙

이 프로젝트의 핵심 모델은 OpenAI/Gemini/Claude/Ollama/Gemma/Llama/Qwen/Hugging Face pretrained weights를 사용하지 않는다.

웹 검색은 최신 정보의 **검색 결과**를 가져오는 보조 기능이며, 검색 결과를 외부 AI에게 보내 생성하도록 만들지 않는다.
