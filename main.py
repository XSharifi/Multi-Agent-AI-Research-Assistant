"""CLI entry point:  python main.py "Your research question" """
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from assistant.app import build_orchestrator
from assistant.llm import LLM


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-agent AI research assistant")
    ap.add_argument("question", help="the research question")
    ap.add_argument("--papers", default="papers", help="folder with PDF papers (default: papers)")
    ap.add_argument("--no-code", action="store_true", help="disable the coding agent")
    ap.add_argument("--rounds", type=int, default=1, help="max evaluator-driven revision rounds (default: 1)")
    ap.add_argument("--provider", help="anthropic | openai (overrides LLM_PROVIDER)")
    ap.add_argument("--model", help="model name (overrides LLM_MODEL)")
    ap.add_argument("--out", default="reports", help="folder for the saved report")
    ap.add_argument("-v", "--verbose", action="store_true", help="show agent steps")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")

    llm = LLM(provider=args.provider, model=args.model)
    orchestrator = build_orchestrator(llm, papers_dir=args.papers, enable_code=not args.no_code,
                                      max_revision_rounds=args.rounds)
    print(f"Agents: {', '.join(orchestrator.agents)}  |  model: {llm.model}\n")

    outcome = orchestrator.run(args.question)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(outcome["report"], encoding="utf-8")

    print(outcome["report"])
    score = outcome["evaluation"].get("overall")
    print(f"\n--- evaluator score: {score}  |  saved to {path} ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
