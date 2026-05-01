"""End-to-end demo: ask 5 sample questions, print answers + citations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from m4_earnings_rag.rag import answer, check_ollama, format_for_terminal


SAMPLE_QUESTIONS = [
    "How does Apple describe iPhone revenue and product trends in its most recent 10-Q?",
    "What does Microsoft say about Azure / cloud growth in its latest MD&A?",
    "What macroeconomic risks does JPMorgan highlight in its most recent 10-Q?",
    "How does NVIDIA characterize data center demand and customer concentration?",
    "What does Tesla say about automotive gross margin pressures?",
]


def main() -> None:
    ok, info = check_ollama()
    print(f"Ollama: {'OK ' + info if ok else 'DOWN — ' + info}")
    for i, q in enumerate(SAMPLE_QUESTIONS, start=1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(SAMPLE_QUESTIONS)}]")
        a = answer(q)
        print(format_for_terminal(a))


if __name__ == "__main__":
    main()
