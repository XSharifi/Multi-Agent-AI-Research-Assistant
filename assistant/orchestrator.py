"""Orchestrator: plan -> dispatch agents in parallel -> evaluate -> (optional revision) -> final report."""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

from .base_agent import AgentResult, BaseAgent
from .evaluation_agent import EvaluationAgent
from .llm import LLM, extract_json

log = logging.getLogger("assistant.orchestrator")

PLAN_SYSTEM = """You are the orchestrator of a multi-agent research assistant.
Break the user's question into sub-tasks and assign each to the most suitable agent.

Available agents:
{agents}

Rules:
- Use only the agents listed above, and as few as the question needs (usually 1 to 3).
- Each task must be self-contained and specific; the agent will not see the other tasks.
- Reply ONLY with JSON: {{"tasks": [{{"agent": "<agent name>", "task": "<instruction>"}}]}}"""

REPORT_SYSTEM = """You are the report writer of a multi-agent research assistant.
Write the final report in Markdown using ONLY the agent outputs provided.
Structure: a title, "Summary" (3-5 sentences), "Findings" (organised by theme, not by agent),
"Code and Experiments" (only if the coding agent produced something), "Sources" (every URL / paper+page cited),
and "Confidence and Limitations" (reflect the evaluator's issues honestly, including the overall score).
Do not add facts, sources or numbers that are not in the agent outputs. If agents contradict each other, say so."""


class Orchestrator:
    def __init__(self, llm: LLM, agents: dict[str, BaseAgent], evaluator: EvaluationAgent, max_revision_rounds: int = 1):
        self.llm = llm
        self.agents = agents
        self.evaluator = evaluator
        self.max_revision_rounds = max_revision_rounds

    # 1. plan ---------------------------------------------------------------
    def plan(self, question: str) -> dict[str, str]:
        listing = "\n".join(f"- {n}: {a.goal}" for n, a in self.agents.items())
        reply = self.llm.chat(PLAN_SYSTEM.format(agents=listing),
                              [{"role": "user", "content": question}], max_tokens=800)
        tasks: dict[str, str] = {}
        try:
            for item in extract_json(reply).get("tasks", []):
                name, task = item.get("agent"), item.get("task")
                if name in self.agents and task:
                    tasks[name] = f"{tasks[name]}\n{task}" if name in tasks else task
        except (ValueError, AttributeError):
            log.warning("Planner returned an unusable plan; falling back to the research agent")
        if not tasks:
            first = "research_agent" if "research_agent" in self.agents else next(iter(self.agents))
            tasks[first] = question
        return tasks

    # 2. dispatch -----------------------------------------------------------
    def _run_agent(self, name: str, task: str, feedback: str = "") -> AgentResult:
        try:
            return self.agents[name].run(task, feedback=feedback)
        except Exception as e:  # one failing agent must not sink the whole run
            log.exception("%s failed", name)
            return AgentResult(name, task, f"ERROR: {type(e).__name__}: {e}")

    def _dispatch(self, jobs: dict[str, tuple[str, str]]) -> dict[str, AgentResult]:
        with ThreadPoolExecutor(max_workers=max(len(jobs), 1)) as pool:
            futures = {n: pool.submit(self._run_agent, n, task, fb) for n, (task, fb) in jobs.items()}
            return {n: f.result() for n, f in futures.items()}

    # 3-5. run --------------------------------------------------------------
    def run(self, question: str) -> dict:
        plan = self.plan(question)
        log.info("Plan: %s", json.dumps(plan)[:300])
        results = self._dispatch({n: (t, "") for n, t in plan.items()})

        evaluation: dict = {}
        for round_no in range(self.max_revision_rounds + 1):
            evaluation = self.evaluator.evaluate(question, results)
            log.info("Evaluation round %d: overall=%s verdict=%s", round_no, evaluation.get("overall"), evaluation.get("verdict"))
            if evaluation.get("verdict") != "revise" or round_no == self.max_revision_rounds:
                break
            followups = {n: fb for n, fb in evaluation.get("followups", {}).items() if n in self.agents and fb}
            if not followups:
                break
            jobs = {n: (results[n].task if n in results else question, fb) for n, fb in followups.items()}
            results.update(self._dispatch(jobs))

        report = self._write_report(question, results, evaluation)
        return {"question": question, "plan": plan, "results": results, "evaluation": evaluation, "report": report}

    def _write_report(self, question: str, results: dict[str, AgentResult], evaluation: dict) -> str:
        material = "\n\n".join(f"## Output of {n}\nTask: {r.task}\n\n{r.output}" for n, r in results.items())
        user = (f"Question: {question}\n\n{material}\n\n"
                f"## Evaluator verdict\n{json.dumps(evaluation, indent=2)}")
        return self.llm.chat(REPORT_SYSTEM, [{"role": "user", "content": user}], max_tokens=3500).strip()
