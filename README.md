#  LLM Fine-Tuner

LoRA(PEFT) 파인튜닝하는 범용 학습 파이프라인입니다.  
Gemma3, EXAONE, Llama3, Qwen, Mistral 등 다양한 모델을 단일 설정 파일(`config.yaml`)로 학습할 수 있습니다.

---

##  구조

```
trainer/
 config.yaml             # 학습 설정 파일 (여기만 수정하면 됨)
 train.py                # 메인 학습 스크립트
 prepare_dataset.py      # 데이터셋 다운로드 및 JSONL 변환
 prepare_dataset.sh      # prepare_dataset.py 실행 쇼트컷
 inference_adapter.py    # [방법 A] 어댑터 분리 로드 추론
 merge_and_export.py     # [방법 B] 어댑터 병합 후 단일 모델 저장
 installFLASHATTN.sh     # Flash Attention 빌드·설치 스크립트
 pyproject.toml          # 프로젝트 의존성 (uv 관리)
 data/                   # 학습 데이터셋 저장 위치
 outputs/                # 학습 결과(체크포인트) 저장 위치
```

---

##  빠른 시작

### 1단계 — 환경 설치

```bash
# uv가 없다면 먼저 설치
curl -LsSf https://astral.sh/uv/install.sh  sh

# 의존성 설치 (uv.lock 기반, 재현 가능)
uv sync
```

 **Python 버전**: 3.12 기준으로 작업되었습니다.

---

### 2단계 — Flash Attention 설치 (권장)

Flash Attention은 CUDA C 소스로부터 빌드해야 합니다.  
빌드 후 학습 속도가 크게 향상되며, VRAM 사용량도 줄어듭니다.

```bash
bash installFLASHATTN.sh
```

 **사전 요구사항**: CUDA Toolkit(`nvcc`), GPU 드라이버가 설치되어 있어야 합니다.  
 빌드 시간: 1030분 이상.. (최초 1회)

Flash Attention을 사용하지 않으려면 `config.yaml`에서 비활성화하면 됩니다.

```yaml
model:
  use_flash_attention: false
```

---

### 3단계 — 데이터셋 준비

샘플 데이터셋(KoAlpaca)을 다운받아 `./data/sample_dataset.jsonl`로 변환합니다.

```bash
bash prepare_dataset.sh
# 또는
uv run python prepare_dataset.py
```

#### 직접 데이터셋을 만들 경우

학습 데이터는 다음 형식의 **JSONL 파일**이어야 합니다.

```jsonl
{"text": "### 질문:\n파이썬이란?\n\n### 답변:\n파이썬은 고수준 프로그래밍 언어입니다."}
{"text": "### 질문:\n...\n\n### 답변:\n..."}
```

- 파일 경로와 텍스트 컬럼명은 `config.yaml`의 `dataset` 섹션에서 지정합니다.

---

### 4단계 — 설정 파일 수정

`config.yaml`을 열어 필요한 항목을 수정합니다.

```yaml
job_name: "my_finetune_v1"                # 작업명 (출력 폴더명에 사용)

model:
  name_or_path: "LGAI-EXAONE/EXAONE-4.0.1-32B"  # HuggingFace 모델 ID 또는 로컬 경로
  use_flash_attention: true               # Flash Attention 사용 여부
  torch_dtype: "bfloat16"                 # 가중치 타입 (bfloat16 권장)

dataset:
  path: "./data/sample_dataset.jsonl"     # JSONL 데이터셋 경로
  text_column: "text"                     # 학습 텍스트가 담긴 키 이름

training:
  output_dir: "./outputs/my_finetune_v1"  # 체크포인트 저장 경로

  per_device_train_batch_size: 16         # GPU 1개당 배치 크기
  gradient_accumulation_steps: 2          # 그래디언트 누적 스텝

  learning_rate: 2.0e-5                   # 학습률
  num_train_epochs: 3                     # 학습 에포크 수
  logging_steps: 10                       # 로그 출력 주기
  max_seq_length: 2048                    # 최대 입력 토큰 길이

  lora_r: 16                              # LoRA 행렬 차원 (어려운 태스크: 3264)
  lora_alpha: 32                          # LoRA 스케일 계수 (보통 lora_r  2)
```

details
summary 주요 파라미터 가이드/summary

 파라미터  권장값  설명 
---------
 `per_device_train_batch_size`  832  클수록 학습 안정, VRAM 소모 증가 
 `gradient_accumulation_steps`  14  VRAM 부족 시 늘려 실효 배치 크기 유지 
 `max_seq_length`  20488192  길수록 긴 문서 처리 가능, VRAM 증가 
 `lora_r`  1664  클수록 표현력 증가, VRAM·파라미터 증가 
 `lora_alpha`  lora_r  2  LoRA 스케일링, 보통 lora_r의 2배 설정 

/details

---

### 5단계 — 학습 실행

```bash
uv run python train.py --config config.yaml
```

학습이 완료되면 `output_dir`에 LoRA 어댑터와 토크나이저가 저장됩니다.

```
outputs/my_finetune_v1/
 adapter_config.json
 adapter_model.safetensors
 tokenizer.json
 ...
```

---

##  추론 (Inference)

학습 완료 후 두 가지 방법으로 추론할 수 있습니다.

### 방법 A — 어댑터 분리 로드 (실험·비교용)

베이스 모델과 LoRA 어댑터를 분리된 상태로 로드합니다.  
어댑터만 교체하며 빠르게 실험하거나 비교할 때 유용합니다.

```bash
uv run python inference_adapter.py \
  --config config.yaml \
  --prompt "파이썬의 장점을 알려줘"
```

 옵션  기본값  설명 
---------
 `--config`  (필수)  학습 시 사용한 config.yaml 경로 
 `--prompt`  (필수)  추론 프롬프트 
 `--adapter_path`  config의 output_dir  어댑터 경로 (기본: config 기준) 
 `--max_new_tokens`  512  최대 생성 토큰 수 

---

### 방법 B — 어댑터 병합 후 단일 모델 저장 (서빙용)

LoRA 어댑터를 베이스 모델에 흡수시켜 완전한 단일 모델로 저장합니다.  
vLLM, TGI, Ollama 등 서빙 인프라에 바로 배포할 때 사용합니다.

```bash
# 기본: output_dir  "_merged" 경로에 저장
uv run python merge_and_export.py --config config.yaml

# 저장 경로 직접 지정
uv run python merge_and_export.py \
  --config config.yaml \
  --output_dir ./outputs/my_merged_model
```

이후 vLLM으로 서빙예시: 

```bash
vllm serve ./outputs/my_merged_model --dtype bfloat16
```

---

##  지원 모델

`model.name_or_path`에 HuggingFace 모델 ID를 입력하면 대부분의 모델을 지원합니다.

 모델 계열  예시 
------
 EXAONE  `LGAI-EXAONE/EXAONE-4.0.1-32B` 
 Gemma 3  `google/gemma-3-12b-it` 
 Llama 3  `meta-llama/Llama-3.1-8B-Instruct` 
 Qwen  `Qwen/Qwen2.5-14B-Instruct` 
 Mistral  `mistralai/Mistral-7B-Instruct-v0.3` 

 **Gemma3 전용**: `token_type_ids` 자동 주입이 내장되어 있어 별도 설정 없이 학습 가능합니다. (토큰 타입별로 0/1/2 구분 존재)

