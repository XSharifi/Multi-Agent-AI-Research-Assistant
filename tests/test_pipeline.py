"""Offline tests: a scripted fake LLM drives the whole pipeline, so no API key or internet is needed.

Run:  python -m tests.test_pipeline      (or: pytest -q)
"""
import json
import os
import tempfile

from assistant.app import build_orchestrator
from assistant.base_agent import parse_reply
from assistant.llm import extract_json
from assistant.tools.pdf_rag import PDFIndex
from assistant.tools.python_runner import run_python


def make_pdf(path, pages):
    """Write a minimal valid PDF (Helvetica text) so the RAG tool can be tested without extra libraries."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        page_id = len(objs) + 1
        kids.append(f"{page_id} 0 R")
        stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_id + 1} 0 R "
                    f"/Resources << /Font << /F1 3 0 R >> >> >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    with open(path, "wb") as f:
        f.write(out)


class ScriptedLLM:
    """Pretends to be an LLM: decides what to say from the system prompt and conversation length."""
    model = "scripted"

    def __init__(self):
        self.eval_calls = 0

    def chat(self, system, messages, max_tokens=0):
        if "orchestrator of a multi-agent" in system:
            return json.dumps({"tasks": [
                {"agent": "research_agent", "task": "Find what a transformer is."},
                {"agent": "paper_agent", "task": "What does the local paper say about attention?"},
                {"agent": "coding_agent", "task": "Compute 6*7 with Python."},
            ]})
        first_turn = len(messages) == 1
        if "You are the research_agent" in system:
            return ("Thought: search.\nAction: web_search\nAction Input: transformer architecture" if first_turn
                    else "Thought: done.\nFinal Answer: Transformers use attention. Source: https://example.com/t")
        if "You are the paper_agent" in system:
            return ("Thought: look in papers.\nAction: search_papers\nAction Input: attention mechanism" if first_turn
                    else "Thought: done.\nFinal Answer: The paper says attention weighs tokens (demo.pdf, p.1).")
        if "You are the coding_agent" in system:
            return ("Thought: compute.\nAction: run_python\nAction Input: ```python\nprint(6*7)\n```" if first_turn
                    else "Thought: done.\nFinal Answer: Code printed 42.")
        if "evaluation agent" in system:
            self.eval_calls += 1
            return json.dumps({"scores": {"relevance": 9}, "overall": 8.5, "issues": ["demo only"],
                               "verdict": "accept", "followups": {}})
        if "report writer" in system:
            return "# Report\n\n## Summary\nTransformers use attention.\n"
        raise AssertionError("unexpected prompt: " + system[:80])


def test_parse_reply():
    assert parse_reply("Thought: x\nAction: web_search\nAction Input: \"abc\"") == ("action", "web_search", "abc")
    assert parse_reply("Thought: x\nFinal Answer: done")[0] == "final"
    assert parse_reply("rambling") is None


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}


def test_python_runner():
    assert "42" in run_python("print(6*7)")
    assert "timed out" in run_python("while True: pass", timeout=1)
    os.environ["FAKE_API_KEY"] = "secret"
    assert "secret" not in run_python("import os; print(os.environ.get('FAKE_API_KEY'))")


def test_pdf_rag():
    with tempfile.TemporaryDirectory() as d:
        make_pdf(os.path.join(d, "demo.pdf"),
                 ["Attention mechanisms weigh the importance of tokens in a sequence.",
                  "Convolutional networks use local filters for images."])
        idx = PDFIndex()
        assert idx.add_folder(d) == 2
        hit = idx.search("attention tokens")
        assert "demo.pdf, p.1" in hit and "p.2" not in hit.split("\n\n")[0]


def test_full_pipeline():
    with tempfile.TemporaryDirectory() as d:
        make_pdf(os.path.join(d, "demo.pdf"), ["Attention mechanisms weigh tokens in a sequence."])
        llm = ScriptedLLM()
        orch = build_orchestrator(llm, papers_dir=d, search_fn=lambda q: "[1] Fake result\nhttps://example.com/t\nTransformers...")
        assert set(orch.agents) == {"research_agent", "paper_agent", "coding_agent"}
        out = orch.run("Explain transformers")
        assert set(out["results"]) == {"research_agent", "paper_agent", "coding_agent"}
        assert "42" in out["results"]["coding_agent"].trace[0] or "42" in out["results"]["coding_agent"].output
        assert out["evaluation"]["overall"] == 8.5 and out["report"].startswith("# Report")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
