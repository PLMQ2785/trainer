# Unsloth QLoRA Trainer

Unsloth, TRL, PEFT를 사용해 여러 계열의 언어 모델을 4-bit QLoRA로
파인튜닝하고, LoRA 어댑터·16-bit 병합 모델·GGUF 모델로 내보내는
워크플로입니다.

## 워크플로

```text
원본 모델
  → 4-bit QLoRA 학습
  → LoRA 어댑터 저장
  → 16-bit 병합 모델 또는 GGUF 양자화 모델 내보내기
```

학습 결과인 LoRA 어댑터는 항상 별도로 보존합니다. 배포 모델은
`merge_and_export.py`에서 필요할 때 생성합니다.

## 구성

- `train.py`: Unsloth QLoRA SFT 학습
- `inference_adapter.py`: 4-bit 베이스 + LoRA 어댑터 추론
- `merge_and_export.py`: 16-bit 병합 또는 GGUF 내보내기
- `prepare_dataset.py`: KoAlpaca 예제 JSONL 생성
- `loop_runner.py`: 여러 설정을 순서대로 반복 학습
- `config.yaml`, `config1.yaml`, `config2.yaml`: 모델별 설정

## 환경 설치

Python 3.12와 CUDA가 설치된 Linux 환경을 기준으로 합니다.

```bash
uv sync
```

Unsloth가 사용하는 PyTorch, Triton, xformers, bitsandbytes는 CUDA 및
드라이버 조합에 민감합니다. 설치 충돌이 있으면 Unsloth 공식 Docker
이미지 또는 설치 가이드의 CUDA별 명령을 우선 사용하세요.

## 데이터 준비

```bash
uv run python prepare_dataset.py
```

기본 데이터 형식은 텍스트 컬럼 하나를 가진 JSONL입니다.

```json
{"text": "### 질문:\n파이썬이란?\n\n### 답변:\n파이썬은 프로그래밍 언어입니다."}
```

## 설정

```yaml
model:
  name_or_path: "meta-llama/Meta-Llama-3-8B"
  load_in_4bit: true
  trust_remote_code: true

dataset:
  path: "./data/sample_dataset.jsonl"
  text_column: "text"

training:
  output_dir: "./outputs/my_model"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  max_seq_length: 2048
  packing: false

  learning_rate: 2.0e-4
  num_train_epochs: 3
  optim: "adamw_8bit"

  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.0
  use_gradient_checkpointing: "unsloth"
```

주요 설정:

- `load_in_4bit: true`: QLoRA. `false`이면 16-bit LoRA로 로드합니다.
- `max_seq_length`: 실제 TRL 토큰 절단 길이로도 전달됩니다.
- `packing`: 짧은 샘플을 한 시퀀스에 묶어 효율을 높입니다.
- `use_gradient_checkpointing: "unsloth"`: VRAM 사용량을 줄입니다.
- `target_modules`: 생략하면 attention 및 MLP의 표준 projection을 사용합니다.

32B 모델은 `per_device_train_batch_size: 1` 또는 `2`부터 시작하고,
남는 VRAM을 확인한 뒤 올리는 편이 안전합니다.

## 학습

```bash
uv run python train.py --config config.yaml
```

`training.output_dir`에는 LoRA 어댑터와 토크나이저가 저장됩니다.

## 어댑터 추론

```bash
uv run python inference_adapter.py \
  --config config.yaml \
  --prompt "파이썬의 장점을 알려줘"
```

다른 체크포인트를 지정하려면 `--adapter_path`를 사용합니다.

## 16-bit 병합

```bash
uv run python merge_and_export.py \
  --config config.yaml \
  --format merged_16bit
```

기본 저장 경로는 어댑터 폴더 이름 뒤에 `_merged_16bit`가 붙습니다.
이 결과를 vLLM이나 SGLang용 기준 모델로 보관할 수 있습니다.

## GGUF 양자화

```bash
uv run python merge_and_export.py \
  --config config.yaml \
  --format gguf \
  --quantization_method q4_k_m
```

기본값은 `q4_k_m`입니다. GGUF 변환 과정에서 llama.cpp 구성 요소를
다운로드하거나 빌드할 수 있으며 추가 디스크 공간이 필요합니다.

## 권장 검증 순서

1. LoRA 어댑터 상태로 프롬프트 결과를 확인합니다.
2. `merged_16bit` 결과가 같은 품질을 유지하는지 확인합니다.
3. GGUF 등 최종 양자화 모델의 품질과 속도를 비교합니다.
4. 학습과 배포에서 동일한 토크나이저 및 채팅 템플릿을 사용합니다.

Gemma 3 4B 이상은 멀티모달 모델입니다. 현재 데이터 파이프라인은
텍스트 전용이므로 이미지 학습이 필요하면 별도의 vision 데이터
콜레이터와 `FastVisionModel` 워크플로가 필요합니다.
