"""
데이터셋 준비 스크립트
"""

# 테스트로 beomi/KoAlpaca-v1.1a → JSONL 변환

import json
import os
from pathlib import Path
from datasets import load_dataset

OUTPUT_DIR = Path("./data")
OUTPUT_FILE = OUTPUT_DIR / "sample_dataset.jsonl"

DATASET_NAME = "beomi/KoAlpaca-v1.1a"

def format_sample(row: dict) -> dict:
    """instruction + output → text 필드로 합치기 (config.yaml의 text_column: "text" 에 맞춤)"""
    instruction = row.get("instruction", "").strip()
    output      = row.get("output", "").strip()

    text = f"### 질문:\n{instruction}\n\n### 답변:\n{output}"
    return {"text": text}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{DATASET_NAME} 다운로드 중...")
    ds = load_dataset(DATASET_NAME, split="train")
    print(f"   총 샘플 수: {len(ds):,}")

    print(f"JSONL 변환 후 저장: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            sample = format_sample(row)
            # 빈 샘플 스킵
            if not sample["text"].strip():
                continue
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 저장 결과 확인
    line_count = sum(1 for _ in open(OUTPUT_FILE, encoding="utf-8"))
    file_size  = os.path.getsize(OUTPUT_FILE) / 1024 / 1024

    print(f"\n완료!")
    print(f"저장 경로  : {OUTPUT_FILE}")
    print(f"샘플 수    : {line_count:,}")
    print(f"파일 크기  : {file_size:.1f} MB")
    print()

    # 첫 샘플 미리보기
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("── 첫 번째 샘플 미리보기 ──────────────────────")
    print(first["text"][:300])
    print("────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
