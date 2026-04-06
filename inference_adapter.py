"""
[방법 A] 어댑터 분리 로드 추론 스크립트
- 베이스 모델 + LoRA 어댑터를 분리 상태로 로드
- 실험/비교/어댑터 교체용
- 사용법: uv run python inference_adapter.py --config config.yaml --prompt "질문을 입력하세요"
"""

import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="학습 시 사용한 config.yaml 경로")
    parser.add_argument("--prompt", type=str, required=True, help="추론에 사용할 프롬프트")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="어댑터 경로 (미입력 시 config의 output_dir 사용)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_model_id  = cfg["model"]["name_or_path"]
    adapter_path   = args.adapter_path or cfg["training"]["output_dir"]
    use_flash_attn = cfg["model"].get("use_flash_attention", True)

    print(f"[A] 베이스 모델 : {base_model_id}")
    print(f"[A] 어댑터 경로 : {adapter_path}")

    # 1. 베이스 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if use_flash_attn else "sdpa",
        trust_remote_code=True,
        device_map="auto",
    )

    # 2. LoRA 어댑터 위에 올리기
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("[A] 어댑터 로드 완료\n")

    # 3. 추론
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    # 입력 부분 제거 후 출력
    generated = output_ids[0][inputs["input_ids"].shape[-1]:]
    print("=== 응답 ===")
    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
