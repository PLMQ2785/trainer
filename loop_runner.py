"""
loop_runner.py — GPU 사용량 유지용 반복 학습 실행기

지정한 config 목록을 순서대로 학습하고, 전부 끝나면 처음부터 다시 반복합니다.
종료: Ctrl+C

사용법:
  uv run python loop_runner.py                        # 기본: CONFIGS 리스트 순서대로 반복
  uv run python loop_runner.py --configs config.yaml config1.yaml
  uv run python loop_runner.py --configs config.yaml  # 단일 config 무한 반복
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime


# 기본 config 목록 (--configs 인자로 덮어쓸 수 있음)
DEFAULT_CONFIGS = [
    "config.yaml",
    "config1.yaml",
    "config2.yaml",
]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_train(config: str) -> int:
    """train.py를 subprocess로 실행하고 종료 코드를 반환합니다."""
    cmd = ["uv", "run", "python", "train.py", "--config", config]
    print(f"\n[{now()}] 시작: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd)

    print("-" * 60)
    if result.returncode == 0:
        print(f"[{now()}] 완료 (config={config})")
    else:
        print(f"[{now()}] 오류 발생 — returncode={result.returncode} (config={config})")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="GPU 사용량 유지용 반복 학습 실행기")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="반복 실행할 config 파일 목록 (기본: config.yaml config1.yaml)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="오류 발생 시 루프 중단 (기본: 오류 무시하고 계속 진행)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=5,
        help="각 실행 사이 대기 시간(초) (기본: 5)",
    )
    args = parser.parse_args()

    configs = args.configs
    iteration = 0

    print(f"[{now()}] loop_runner 시작")
    print(f"  configs     : {configs}")
    print(f"  stop-on-error: {args.stop_on_error}")
    print(f"  delay       : {args.delay}s")
    print("  종료하려면 Ctrl+C")
    print("=" * 60)

    try:
        while True:
            iteration += 1
            print(f"\n{'=' * 60}")
            print(f"[{now()}] === 반복 #{iteration} 시작 ===")
            print(f"{'=' * 60}")

            for config in configs:
                returncode = run_train(config)

                if returncode != 0 and args.stop_on_error:
                    print(f"[{now()}] --stop-on-error 설정으로 루프를 중단합니다.")
                    sys.exit(returncode)

                if args.delay > 0:
                    print(f"[{now()}] {args.delay}초 대기 후 다음 실행...")
                    time.sleep(args.delay)

    except KeyboardInterrupt:
        print(f"\n[{now()}] Ctrl+C 감지 — 루프를 종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()
