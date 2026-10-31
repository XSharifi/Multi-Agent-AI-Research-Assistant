"""The three worker agents. Each one is a BaseAgent with its own goal and tools."""
from .base_agent import BaseAgent


class ResearchAgent(BaseAgent):
    name = "research_agent"
    goal = (
        "Find current, reliable information on the web. Run several differently worded "
        "searches, open the most promising pages, cross-check claims across sources, "
        "and cite every source by URL."
    )


class PaperAgent(BaseAgent):
    name = "paper_agent"
    goal = (
        "Answer questions using the user's local PDF papers. Retrieve passages with "
        "search_papers, quote or paraphrase them accurately, and cite each one as "
        "(paper file name, page number). Do not use knowledge outside the retrieved passages."
    )


class CodingAgent(BaseAgent):
    name = "coding_agent"
    goal = (
        "Write small, self-contained Python programs that compute, simulate or verify "
        "things needed for the task, then run them with run_python and read the output. "
        "Only the Python standard library is guaranteed to be available. Always print "
        "results. In your Final Answer include the final code and the real output you observed."
    )
