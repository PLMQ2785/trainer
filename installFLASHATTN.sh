#!/usr/bin/env bash
# =============================================================================
# Flash Attention Build & Install Script
# =============================================================================
# Flash Attention은 C++/CUDA 확장을 컴파일해야 하므로
# 빌드 환경(CUDA toolkit, ninja, packaging 등)이 미리 갖춰져야 합니다.
# 최종 설치 명령: uv pip install -v flash-attn --no-build-isolation
# =============================================================================

set -euo pipefail

# ── 색상 출력 헬퍼 ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 병렬 빌드 잡 수 (메모리 부족 시 줄이세요) ─────────────────────────────────
MAX_JOBS=${MAX_JOBS:-4}

# =============================================================================
# 1. 사전 의존성 확인
# =============================================================================
info "=== Flash Attention 빌드 환경 확인 ==="

# uv 확인
if ! command -v uv &>/dev/null; then
    error "uv 가 설치되어 있지 않습니다. https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi
success "uv: $(uv --version)"

# Python 확인
PYTHON=$(uv run python -c "import sys; print(sys.executable)" 2>/dev/null || python3 -c "import sys; print(sys.executable)")
PYTHON_VER=$(${PYTHON} --version)
success "Python: ${PYTHON_VER} (${PYTHON})"

# nvcc / CUDA 확인
if ! command -v nvcc &>/dev/null; then
    error "nvcc 를 찾을 수 없습니다. CUDA Toolkit이 설치되어 있는지, PATH에 포함되어 있는지 확인하세요."
    error "예: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi
CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+")
success "CUDA: ${CUDA_VER} ($(which nvcc))"

# ninja 확인
if ! command -v ninja &>/dev/null; then
    warn "ninja 빌드 툴이 PATH에 없습니다. 빌드 속도가 느릴 수 있습니다."
    warn "설치: uv pip install ninja"
else
    success "ninja: $(ninja --version)"
fi

# PyTorch 및 CUDA 연동 확인
info "PyTorch CUDA 연동 확인 중..."
${PYTHON} - <<'PYCHECK'
import sys
try:
    import torch
    print(f"  torch       : {torch.__version__}")
    print(f"  CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version   : {torch.version.cuda}")
        print(f"  GPU            : {torch.cuda.get_device_name(0)}")
    else:
        print("  [WARN] torch.cuda.is_available() == False")
        print("         CUDA가 감지되지 않으면 Flash Attention 빌드가 실패할 수 있습니다.")
except ImportError:
    print("[ERROR] PyTorch가 설치되어 있지 않습니다. 먼저 'uv sync' 를 실행하세요.")
    sys.exit(1)
PYCHECK

# packaging 확인 (--no-build-isolation 시 필수)
${PYTHON} -c "import packaging" 2>/dev/null \
    && success "packaging: OK" \
    || { error "packaging 모듈이 없습니다. 먼저 'uv pip install packaging' 을 실행하세요."; exit 1; }

# =============================================================================
# 2. 환경변수 설정
# =============================================================================
info "=== 빌드 환경변수 설정 ==="

# CUDA_HOME 자동 감지
if [[ -z "${CUDA_HOME:-}" ]]; then
    CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
    export CUDA_HOME
fi
success "CUDA_HOME=${CUDA_HOME}"

# MAX_JOBS: 병렬 컴파일 워커 수
export MAX_JOBS
success "MAX_JOBS=${MAX_JOBS}"

# Flash Attention 빌드 옵션
# FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE  → CUDA 커널 빌드 포함 (기본값)
export FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE

# 필요하면 특정 GPU 아키텍처만 빌드해 시간 단축
# (예: A100=sm_80, H100=sm_90, RTX 4090=sm_89, RTX 3090=sm_86)
# 여러 GPU 아키텍처가 필요하면 세미콜론으로 구분
# 미설정 시 PyTorch가 감지한 아키텍처로 자동 결정됨
# export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"

# =============================================================================
# 3. Flash Attention 빌드 및 설치
# =============================================================================
info "=== Flash Attention 빌드 및 설치 시작 ==="
info "명령어: uv pip install -v flash-attn --no-build-isolation"
info "MAX_JOBS=${MAX_JOBS} / CUDA_HOME=${CUDA_HOME}"
warn "빌드 시간이 상당히 걸릴 수 있습니다 (10~30분). 잠시 기다려 주세요..."

uv pip install -v flash-attn --no-build-isolation

# =============================================================================
# 4. 설치 확인
# =============================================================================
info "=== 설치 검증 ==="
${PYTHON} - <<'PYCHECK'
try:
    import flash_attn
    print(f"  flash_attn version : {flash_attn.__version__}")
    print("[OK] Flash Attention 설치 성공!")
except ImportError as e:
    print(f"[ERROR] Flash Attention import 실패: {e}")
    raise SystemExit(1)
PYCHECK

success "=== 완료! Flash Attention이 성공적으로 설치되었습니다. ==="
