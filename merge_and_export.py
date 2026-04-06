"""
[방법 B] 어댑터 병합 및 단일 모델 저장 스크립트
- LoRA 어댑터를 베이스 모델에 흡수시켜 완전한 단일 모델로 저장
- vLLM / TGI / Ollama 등 서빙 인프라 배포용
- 사용법: uv run python merge_and_export.py --config config.yaml
- 옵션  : --output_dir ./outputs/merged_model  (기본: output_dir + "_merged")
"""

import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="학습 시 사용한 config.yaml 경로")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="어댑터 경로 (미입력 시 config의 output_dir 사용)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="병합 모델 저장 경로 (미입력 시 output_dir + '_merged')")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model_id = cfg["model"]["name_or_path"]
    adapter_path  = args.adapter_path or cfg["training"]["output_dir"]
    output_dir    = args.output_dir   or str(Path(adapter_path) / ".." / (Path(adapter_path).name + "_merged"))
    output_dir    = str(Path(output_dir).resolve())
    use_flash_attn = cfg["model"].get("use_flash_attention", True)

    print(f"[B] 베이스 모델  : {base_model_id}")
    print(f"[B] 어댑터 경로  : {adapter_path}")
    print(f"[B] 저장 경로    : {output_dir}")
    print("[B] 병합 중... (메모리를 많이 사용합니다)")

    # 1. 베이스 모델 + 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        trust_remote_code=True,
        device_map="auto",
    )

    # 2. 어댑터 로드
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # 3. 어댑터를 베이스에 흡수 (merge_and_unload)
    #    - LoRA 가중치가 원본 W에 더해져 단일 텐서가 됨
    #    - PEFT 의존성 없이 일반 transformers 모델로 사용 가능
    print("[B] merge_and_unload 실행 중...")
    merged_model = peft_model.merge_and_unload()

    # 4. 저장
    print(f"[B] 저장 중: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✅ 병합 완료!")
    print(f"   저장 경로: {output_dir}")
    print(f"   이 폴더를 vLLM / TGI / Ollama 등에 바로 사용할 수 있습니다.")
    print(f"\n   vLLM 예시:")
    print(f"   vllm serve {output_dir} --dtype bfloat16")


if __name__ == "__main__":
    main()
