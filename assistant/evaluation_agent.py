"""Evaluation Agent: critiques the combined worker outputs before the report is written."""
import logging

from .llm import LLM, extract_json

log = logging.getLogger("assistant.evaluation")

SYSTEM = """You are the evaluation agent in a multi-agent research assistant.
You receive the user's question and the outputs of several worker agents.
Judge them strictly:
- relevance: does the material answer the question?
- evidence: are claims backed by cited sources or executed code output?
- consistency: do the agents agree with each other? Flag contradictions.
- completeness: what important parts of the question are not covered?
Penalise unsupported claims, invented-looking citations, and code results that were not actually executed.

Reply ONLY with a JSON object:
{
  "scores": {"relevance": 0-10, "evidence": 0-10, "consistency": 0-10, "completeness": 0-10},
  "overall": 0-10,
  "issues": ["specific problem", ...],
  "verdict": "accept" or "revise",
  "followups": {"<agent_name>": "what exactly this agent should redo or add", ...}
}
Use "revise" only if a concrete follow-up would materially improve the answer.
Only use agent names that appear in the input."""


class EvaluationAgent:
    name = "evaluation_agent"

    def __init__(self, llm: LLM):
        self.llm = llm

    def evaluate(self, question: str, results: dict) -> dict:
        blocks = "\n\n".join(
            f"### {name}\nTask: {r.task}\n\n{r.output[:6000]}" for name, r in results.items()
        )
        reply = self.llm.chat(SYSTEM, [{"role": "user", "content": f"Question: {question}\n\n{blocks}"}], max_tokens=1200)
        try:
            ev = extract_json(reply)
        except ValueError:
            log.warning("Evaluator returned invalid JSON; accepting by default")
            return {"scores": {}, "overall": None, "issues": ["Evaluator output could not be parsed."],
                    "verdict": "accept", "followups": {}}
        ev.setdefault("issues", [])
        ev.setdefault("verdict", "accept")
        ev.setdefault("followups", {})
        return ev
