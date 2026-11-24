#!/usr/bin/env python3
"""Configure the existing Linux/Python 3.11 GLM-Voice virtual environment.
Run from the project root: python setup_glm_server.py
Does not download model weights or initialize models.
"""
from pathlib import Path
from importlib.metadata import version
import os
import platform
import subprocess
import sys

def ensure_sqlite():
    probe = subprocess.run([
        sys.executable, "-c",
        "import sqlite3; assert sqlite3.sqlite_version_info >= (3, 35, 0); "
        "print('SQLite:', sqlite3.sqlite_version)",
    ], capture_output=True, text=True)
    if probe.returncode == 0:
        print(probe.stdout.strip(), flush=True)
        return
    print("Installing a SQLite replacement in this virtual environment.", flush=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--only-binary=:all:",
        "pysqlite3-binary==0.5.4",
    ], check=True)
    import sysconfig
    site_dir = Path(sysconfig.get_path("purelib")).resolve()
    if not site_dir.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Refusing to write outside the active virtual environment.")
    shim = site_dir / "glm_voice_sqlite.pth"
    shim.write_text(
        "import sys, pysqlite3, pysqlite3._sqlite3; "
        "sys.modules['sqlite3'] = pysqlite3; "
        "sys.modules['sqlite3.dbapi2'] = pysqlite3.dbapi2; "
        "sys.modules['_sqlite3'] = pysqlite3._sqlite3\n",
        encoding="utf-8",
    )
    subprocess.run([
        sys.executable, "-c",
        "import sqlite3; assert sqlite3.sqlite_version_info >= (3, 35, 0); "
        "c=sqlite3.connect(':memory:'); "
        "assert c.execute('select 1').fetchone() == (1,); c.close(); "
        "print('SQLite replacement passed:', sqlite3.sqlite_version)",
    ], check=True)
    print(f"SQLite virtual environment shim: {shim}", flush=True)


def main():
    root = Path.cwd()
    requirements = root / "requirements-server.txt"
    constraints = root / "constraints-server.txt"
    if not (root / "src/spoken_chatbot/model.py").is_file() or not all(
        path.is_file() for path in (requirements, constraints)
    ):
        raise SystemExit("Run this script from the GLM-Voice-RAG-main project directory.")
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This setup targets Linux x86_64; run it on the server.")
    if sys.version_info[:2] != (3, 11) or sys.prefix == sys.base_prefix:
        raise SystemExit("Activate the project's Python 3.11 .venv first.")
    import torch
    expected = {
        "torch": "2.6.0+cu124", "torchaudio": "2.6.0+cu124",
        "fairseq2": "0.4.5", "fairseq2n": "0.4.5+cu124", "sonar-space": "0.4.0",
    }
    for package, wanted in expected.items():
        actual = version(package)
        if actual != wanted:
            raise SystemExit(f"Expected {package} {wanted}, found {actual}; no installation performed.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; no installation performed.")
    ensure_sqlite()
    print(f"Installing remaining dependencies with {sys.executable}", flush=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements), "-c", str(constraints),
    ], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    env = os.environ.copy()
    additions = [str(root / "src"), str(root / "third_party/Matcha-TTS")]
    env["PYTHONPATH"] = os.pathsep.join(additions + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    subprocess.run([sys.executable, "-c", """
import torch
import fairseq2
from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from spoken_chatbot.model import GLM_Voice
from rag_module.e2e_rag import RAG, E2E_RAG, ASR_RAG
import cosyvoice.flow.flow
import cosyvoice.hifigan.generator
print('Project import checks passed; PyTorch:', torch.__version__)
print('No model weights have been downloaded. Model download and GPU allocation are next.')
"""], check=True, env=env)
    print("SERVER ENVIRONMENT SETUP PASSED", flush=True)


if __name__ == "__main__":
    main()
