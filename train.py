import argparse
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


# token_type_ids를 학습 시 필수로 요구하는 모델 목록
# (Gemma3는 Vision-Language 모델이라 텍스트/이미지 토큰 구분용으로 필요)
_MODELS_REQUIRING_TOKEN_TYPE_IDS = frozenset({"gemma3"})


class MultiModelSFTTrainer(SFTTrainer):
    """
    다양한 모델을 범용 지원하는 SFTTrainer.
    token_type_ids 등 모델별 특수 입력이 필요한 경우 자동으로 처리합니다.
    - Gemma3: token_type_ids (0=텍스트, 1=이미지) 필수 → 텍스트 학습 시 zeros 주입
    - Llama/Qwen/Mistral 등: 추가 입력 불필요, 건드리지 않음
    """
    def training_step(self, model, inputs, num_items_in_batch=None):
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        if model_type in _MODELS_REQUIRING_TOKEN_TYPE_IDS:
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        return super().training_step(model, inputs, num_items_in_batch)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print(f"========== [{cfg['job_name']}] ==========")
    print(f"Target Model: {cfg['model']['name_or_path']}")

    # 1. 범용 토크나이저 로드 (trust_remote_code 적용)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg['model']['name_or_path'],
        trust_remote_code=True
    )
    
    # Pad 토큰이 없는 모델(Llama3, Gemma 등)을 위한 방어 로직
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # TRL 1.0: max_seq_length는 SFTConfig에 없음 → tokenizer.model_max_length으로 설정
    tokenizer.model_max_length = cfg['training'].get('max_seq_length', 2048)

    # 2. 범용 모델 로드 (H100/H200 최적화 및 trust_remote_code 적용)
    model = AutoModelForCausalLM.from_pretrained(
        cfg['model']['name_or_path'],
        dtype=torch.bfloat16,  # bfloat16 (torch_dtype은 TRL 1.0+ deprecated)
        attn_implementation="flash_attention_2" if cfg['model'].get('use_flash_attention', True) else "sdpa",
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Pad 토큰을 새로 추가했다면 모델 임베딩 사이즈도 늘려줍니다.
    if tokenizer.pad_token == '[PAD]':
         model.resize_token_embeddings(len(tokenizer))

    # 3. 데이터셋 로드
    dataset = load_dataset("json", data_files=cfg['dataset']['path'], split="train")

    # 4. 범용 PEFT(LoRA) 설정: 어떤 모델이든 'all-linear'로 자동 매핑
    peft_config = LoraConfig(
        r=cfg['training'].get('lora_r', 16),
        lora_alpha=cfg['training'].get('lora_alpha', 32),
        lora_dropout=0.05,
        target_modules="all-linear", # 핵심: 모델 아키텍처 상관없이 모든 Linear 레이어 타겟팅
        task_type="CAUSAL_LM",
    )

    # 5. SFTConfig (TRL 1.0: dataset_text_field는 SFTConfig에, max_seq_length는 tokenizer로 설정)
    sft_config = SFTConfig(
        output_dir=cfg['training']['output_dir'],
        per_device_train_batch_size=cfg['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=cfg['training']['gradient_accumulation_steps'],
        learning_rate=float(cfg['training']['learning_rate']),
        num_train_epochs=cfg['training']['num_train_epochs'],
        logging_steps=cfg['training'].get('logging_steps', 10),
        bf16=True,
        save_strategy="epoch",
        report_to="none",
        dataset_text_field=cfg['dataset']['text_column'],  # SFTConfig에서 관리
    )

    # 6. Trainer 초기화 및 실행
    trainer = MultiModelSFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(cfg['training']['output_dir'])
    tokenizer.save_pretrained(cfg['training']['output_dir']) # 토크나이저도 함께 저장 필수
    print("✅ 학습 및 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()