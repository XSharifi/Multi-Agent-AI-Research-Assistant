"""Python execution tool for the Coding Agent.

WARNING: this runs model-written code in a subprocess with a timeout and a
scrubbed environment (API keys removed). That is NOT a real sandbox. For
anything beyond personal experiments, run the whole assistant inside Docker
or use a dedicated sandbox service.
"""
import os
import re
import subprocess
import sys
import tempfile


def _strip_fences(code: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.S)
    return (m.group(1) if m else code).strip()


def run_python(code: str, timeout: int = 20) -> str:
    code = _strip_fences(code)
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET"))}
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "snippet.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, path], cwd=workdir, env=env,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Execution timed out after {timeout}s."
    return (
        f"exit code: {proc.returncode}\n"
        f"STDOUT:\n{proc.stdout[-3000:]}\n"
        f"STDERR:\n{proc.stderr[-2000:]}"
    )
