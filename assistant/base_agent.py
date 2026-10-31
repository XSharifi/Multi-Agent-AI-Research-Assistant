"""ReAct-style agent loop shared by the Research, Paper and Coding agents.

The loop uses a plain-text protocol (Thought / Action / Action Input /
Final Answer) instead of provider-specific tool calling, so it works with any
LLM backend, including small local models.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from .llm import LLM

log = logging.getLogger("assistant.agent")

MAX_OBSERVATION_CHARS = 4000

_FINAL_RE = re.compile(r"Final Answer:\s*(.*)", re.S | re.I)
_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]\w*)\s*\n\s*Action Input:\s*(.*)", re.S | re.I)


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[str], str]


@dataclass
class AgentResult:
    agent: str
    task: str
    output: str
    trace: list[str] = field(default_factory=list)


def _clean_input(raw: str) -> str:
    raw = raw.strip()
    fence = re.match(r"```\w*\s*\n(.*?)```", raw, re.S)
    if fence:
        return fence.group(1).strip()
    if "\n" not in raw and len(raw) > 1 and raw[0] == raw[-1] and raw[0] in "\"'":
        return raw[1:-1]
    return raw


def parse_reply(reply: str):
    """Return ('action', tool, input), ('final', answer) or None."""
    action, final = _ACTION_RE.search(reply), _FINAL_RE.search(reply)
    if action and (not final or action.start() < final.start()):
        return ("action", action.group(1), _clean_input(action.group(2)))
    if final:
        return ("final", final.group(1).strip())
    return None


class BaseAgent:
    name = "agent"
    goal = "Complete the task you are given."

    def __init__(self, llm: LLM, tools: list[Tool], max_steps: int = 6):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    def system_prompt(self) -> str:
        tool_lines = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
        return (
            f"You are the {self.name}, part of a multi-agent research assistant.\n"
            f"{self.goal}\n\n"
            f"Tools available:\n{tool_lines}\n\n"
            "Work one step at a time using EXACTLY this format:\n"
            "Thought: <your reasoning>\n"
            "Action: <one tool name from the list>\n"
            "Action Input: <the input for the tool>\n"
            "Then stop and wait for the Observation. Do not write the Observation yourself.\n\n"
            "When you have enough information, finish with:\n"
            "Thought: I now have enough information.\n"
            "Final Answer: <your answer in Markdown, with sources cited>\n\n"
            "Rules: never invent sources, quotes or results. If the tools return nothing "
            "useful, say so in your Final Answer."
        )

    def run(self, task: str, feedback: str = "") -> AgentResult:
        content = f"Task: {task}"
        if feedback:
            content += f"\n\nA reviewer asked you to improve your previous attempt. Address this:\n{feedback}"
        messages = [{"role": "user", "content": content}]
        trace: list[str] = []

        for step in range(1, self.max_steps + 1):
            reply = self.llm.chat(self.system_prompt(), messages, max_tokens=2500)
            reply = re.split(r"\n\s*Observation:", reply, maxsplit=1)[0].strip()  # drop hallucinated observations
            trace.append(reply)
            messages.append({"role": "assistant", "content": reply or "(empty)"})

            parsed = parse_reply(reply)
            if parsed is None:
                messages.append({"role": "user", "content": (
                    "Observation: Invalid format. Reply with 'Action:' and 'Action Input:' "
                    "or with 'Final Answer:'.")})
                continue
            if parsed[0] == "final":
                log.info("%s finished in %d step(s)", self.name, step)
                return AgentResult(self.name, task, parsed[1], trace)

            _, tool_name, tool_input = parsed
            log.info("%s step %d -> %s(%.60s)", self.name, step, tool_name, tool_input.replace("\n", " "))
            observation = self._call_tool(tool_name, tool_input)
            messages.append({"role": "user", "content": f"Observation: {observation[:MAX_OBSERVATION_CHARS]}"})

        # Step budget exhausted: force a conclusion from what was gathered.
        messages.append({"role": "user", "content": (
            "Step limit reached. Write your Final Answer now using only what you have gathered.")})
        reply = self.llm.chat(self.system_prompt(), messages, max_tokens=2500)
        trace.append(reply)
        parsed = parse_reply(reply)
        output = parsed[1] if parsed and parsed[0] == "final" else reply.strip()
        return AgentResult(self.name, task, output, trace)

    def _call_tool(self, name: str, arg: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available tools: {', '.join(self.tools)}"
        try:
            return str(tool.fn(arg))
        except Exception as e:
            return f"Tool error: {type(e).__name__}: {e}"
