"""结构化输出示例：python -m scripts.structured_demo"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_agent.structured import extract_contract

DEFAULT_CONTRACT = Path(__file__).resolve().parents[1] / "examples" / "sample_contract.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="合同字段抽取（Pydantic + function_calling）")
    parser.add_argument("--file", default=str(DEFAULT_CONTRACT))
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    result = extract_contract(text)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
