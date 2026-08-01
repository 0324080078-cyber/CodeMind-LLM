"""Secure Code Execution Sandbox."""

import os
import sys
import time
import subprocess
import tempfile
from typing import Dict, Any, Optional

LANGUAGE_CONFIGS = {
    "python":     {"ext": ".py",   "cmd": [sys.executable]},
    "javascript": {"ext": ".js",   "cmd": ["node"]},
    "typescript": {"ext": ".ts",   "cmd": ["ts-node"]},
    "bash":       {"ext": ".sh",   "cmd": ["bash"]},
    "go":         {"ext": ".go",   "cmd": ["go", "run"]},
    "ruby":       {"ext": ".rb",   "cmd": ["ruby"]},
    "php":        {"ext": ".php",  "cmd": ["php"]},
    "rust":       {"ext": ".rs",   "cmd": ["rustc", "-o", "/tmp/cm_rust_out"]},
}

BLOCKED = ["os.system(", "subprocess.call(", "__import__('os').system", "shutil.rmtree('/')"]


class CodeSandbox:
    def __init__(self, timeout=30, max_memory_mb=512, max_output=65536):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_output = max_output

    def execute(self, code, language="python", stdin=None):
        for pat in BLOCKED:
            if pat in code:
                return {"stdout": "", "stderr": f"Blocked pattern: {pat}", "exit_code": -1, "execution_time": 0, "blocked": True}

        lang = LANGUAGE_CONFIGS.get(language.lower())
        if not lang:
            return {"stdout": "", "stderr": f"Language not supported: {language}", "exit_code": -1, "execution_time": 0}

        with tempfile.NamedTemporaryFile(mode="w", suffix=lang["ext"], delete=False, dir="/tmp") as f:
            f.write(code)
            tmp = f.name

        try:
            cmd = lang["cmd"] + [tmp]
            t0 = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE if stdin else None)
            try:
                out, err = proc.communicate(input=stdin.encode() if stdin else None, timeout=self.timeout)
                code_rc = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = b"", b"Timed out."
                code_rc = -1
            elapsed = time.time() - t0
            return {"stdout": out.decode("utf-8", errors="replace")[:self.max_output], "stderr": err.decode("utf-8", errors="replace")[:4096], "exit_code": code_rc, "execution_time": round(elapsed, 3), "language": language, "blocked": False}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "exit_code": -1, "execution_time": 0, "blocked": False}
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass
