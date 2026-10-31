"""Wires tools, agents and the orchestrator together."""
from pathlib import Path

from .agents import CodingAgent, PaperAgent, ResearchAgent
from .base_agent import Tool
from .evaluation_agent import EvaluationAgent
from .llm import LLM
from .orchestrator import Orchestrator
from .tools.pdf_rag import PDFIndex
from .tools.python_runner import run_python
from .tools.web_search import read_url, web_search


def build_orchestrator(llm: LLM, papers_dir: str | None = "papers", enable_code: bool = True,
                       search_fn=web_search, max_revision_rounds: int = 1) -> Orchestrator:
    agents = {
        "research_agent": ResearchAgent(llm, [
            Tool("web_search", "Search the web. Input: a search query.", search_fn),
            Tool("read_url", "Read the text of a web page. Input: a full URL.", read_url),
        ]),
    }

    index = PDFIndex()
    if papers_dir and Path(papers_dir).is_dir():
        index.add_folder(papers_dir)
    if index.chunks:  # only offer the Paper Agent when there is something to read
        agents["paper_agent"] = PaperAgent(llm, [
            Tool("search_papers", "Search the local PDF papers. Input: a natural-language query.", index.search),
            Tool("list_papers", "List indexed PDF file names. Input: leave empty.", index.list_papers),
        ])

    if enable_code:
        agents["coding_agent"] = CodingAgent(llm, [
            Tool("run_python", "Run Python code and return stdout/stderr. Input: the code.", run_python),
        ], max_steps=6)

    return Orchestrator(llm, agents, EvaluationAgent(llm), max_revision_rounds=max_revision_rounds)
