"""Unsloth 기반 범용 QLoRA SFT 학습 스크립트."""

import argparse
from pathlib import Path

import yaml
from unsloth import FastModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unsloth QLoRA SFT trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    model_name = model_cfg["name_or_path"]
    output_dir = Path(training_cfg["output_dir"])
    max_seq_length = int(training_cfg.get("max_seq_length", 2048))
    seed = int(training_cfg.get("seed", 3407))
    load_in_4bit = bool(model_cfg.get("load_in_4bit", True))

    print(f"========== [{cfg['job_name']}] ==========")
    print(f"Target Model : {model_name}")
    print(f"Training     : {'QLoRA (4-bit)' if load_in_4bit else 'LoRA (16-bit)'}")
    print(f"Adapter Path : {output_dir}")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_16bit=not load_in_4bit,
        full_finetuning=False,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )

    target_modules = training_cfg.get("target_modules", DEFAULT_TARGET_MODULES)
    model = FastModel.get_peft_model(
        model,
        r=int(training_cfg.get("lora_r", 16)),
        target_modules=target_modules,
        lora_alpha=int(training_cfg.get("lora_alpha", 32)),
        lora_dropout=float(training_cfg.get("lora_dropout", 0.0)),
        bias="none",
        use_gradient_checkpointing=training_cfg.get(
            "use_gradient_checkpointing", "unsloth"
        ),
        random_state=seed,
        use_rslora=bool(training_cfg.get("use_rslora", False)),
        loftq_config=None,
    )

    dataset = load_dataset(
        "json",
        data_files=dataset_cfg["path"],
        split="train",
    )

    bf16 = is_bfloat16_supported()
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(
            training_cfg["per_device_train_batch_size"]
        ),
        gradient_accumulation_steps=int(
            training_cfg["gradient_accumulation_steps"]
        ),
        learning_rate=float(training_cfg["learning_rate"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "linear"),
        optim=training_cfg.get("optim", "adamw_8bit"),
        bf16=bf16,
        fp16=not bf16,
        max_length=max_seq_length,
        packing=bool(training_cfg.get("packing", False)),
        dataset_text_field=dataset_cfg["text_column"],
        dataset_num_proc=training_cfg.get("dataset_num_proc"),
        save_strategy=training_cfg.get("save_strategy", "epoch"),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        report_to=training_cfg.get("report_to", "none"),
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=sft_config,
    )

    trainer.train(
        resume_from_checkpoint=training_cfg.get("resume_from_checkpoint")
    )

    # 작은 LoRA 어댑터만 저장합니다. 16-bit 병합과 최종 양자화는 별도 수행합니다.
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"✅ QLoRA 학습 완료: {output_dir}")


if __name__ == "__main__":
    main()
