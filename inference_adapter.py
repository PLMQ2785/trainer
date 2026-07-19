"""Unsloth로 베이스 모델과 QLoRA 어댑터를 로드해 추론합니다."""

import argparse

import torch
import yaml
from unsloth import FastModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="학습에 사용한 config 경로")
    parser.add_argument("--prompt", required=True, help="추론 프롬프트")
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="어댑터 경로 (기본: config의 output_dir)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    adapter_path = args.adapter_path or cfg["training"]["output_dir"]
    max_seq_length = int(cfg["training"].get("max_seq_length", 2048))

    print(f"어댑터 로드: {adapter_path}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=bool(cfg["model"].get("load_in_4bit", True)),
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    print("=== 응답 ===")
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
