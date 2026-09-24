import argparse
from generate_final import load_bundle, generate


TESTS = [
    "돌이는 누구야?",
    "돌이의 특징은 무엇이야?",
    "돌이는 어떤 동물이야?",
    "대한민국의 수도는 어디야?",
    "3 더하기 5는 얼마야?",
    "1 더하기 2는 얼마야?",
    "파이썬이란 무엇이야?",
    "체스에서 킹은 어떻게 움직여?",
    "지구는 어떤 행성이야?",
    "안녕",
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best.npz",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    print("=" * 70)
    print("DORI AI MODEL EVALUATION")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Tokens     : {args.tokens}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-K      : {args.top_k}")
    print(f"Top-P      : {args.top_p}")
    print("=" * 70)

    print("\n모델을 불러오는 중...")

    tok, model = load_bundle(args.checkpoint)

    print("모델 로드 완료.")
    print(f"Vocab size : {tok.vocab_size}")
    print(f"Context    : {model.max_len}")
    print()

    for i, question in enumerate(TESTS, 1):
        print("-" * 70)
        print(f"[TEST {i}/{len(TESTS)}]")
        print(f"질문: {question}")

        try:
            answer = generate(
                question,
                tok,
                model,
                tokens=args.tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=1000 + i,
            )

            print(f"답변: {answer}")

        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")

    print("-" * 70)
    print("평가 완료.")


if __name__ == "__main__":
    main()