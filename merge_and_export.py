"""QLoRA 어댑터를 16-bit 모델 또는 GGUF로 내보냅니다."""

import argparse
from pathlib import Path

import yaml
from unsloth import FastModel


def default_output_dir(adapter_path: str, export_format: str, quant: str) -> str:
    adapter = Path(adapter_path)
    suffix = "merged_16bit" if export_format == "merged_16bit" else f"gguf_{quant}"
    return str((adapter.parent / f"{adapter.name}_{suffix}").resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="학습에 사용한 config 경로")
    parser.add_argument(
        "--adapter_path",
        default=None,
        help="어댑터 경로 (기본: config의 output_dir)",
    )
    parser.add_argument("--output_dir", default=None, help="내보낼 모델 경로")
    parser.add_argument(
        "--format",
        choices=("merged_16bit", "gguf"),
        default="merged_16bit",
        help="내보내기 형식 (기본: merged_16bit)",
    )
    parser.add_argument(
        "--quantization_method",
        default="q4_k_m",
        help="GGUF 양자화 방식 (기본: q4_k_m)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    adapter_path = args.adapter_path or cfg["training"]["output_dir"]
    output_dir = args.output_dir or default_output_dir(
        adapter_path, args.format, args.quantization_method
    )
    max_seq_length = int(cfg["training"].get("max_seq_length", 2048))

    print(f"어댑터 경로 : {adapter_path}")
    print(f"내보내기 형식: {args.format}")
    print(f"저장 경로    : {output_dir}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=bool(cfg["model"].get("trust_remote_code", True)),
    )

    if args.format == "merged_16bit":
        model.save_pretrained_merged(
            output_dir,
            tokenizer,
            save_method="merged_16bit",
        )
    else:
        model.save_pretrained_gguf(
            output_dir,
            tokenizer,
            quantization_method=args.quantization_method,
        )

    print(f"✅ 내보내기 완료: {output_dir}")


if __name__ == "__main__":
    main()
