#!/usr/bin/env python3
"""Configure the existing Linux/Python 3.11 GLM-Voice virtual environment.
Run from the project root: python setup_glm_server.py
Does not download model weights or initialize models.
"""
from pathlib import Path
from importlib.metadata import version
import ast
import os
import platform
import shutil
import subprocess
import sys

REQUIREMENTS = 'conformer==0.3.2\ndiffusers==0.27.2\nfastapi==0.115.3\nfastapi-cli==0.0.4\ngdown==5.1.0\nhuggingface_hub==0.25.2\nhydra-core==1.3.2\nHyperPyYAML==1.2.2\ninflect==7.3.1\nlibrosa==0.10.2\nlightning==2.2.4\nmatplotlib==3.7.5\nmodelscope==1.15.0\nnetworkx==3.1\nnumpy==1.24.4\nomegaconf==2.3.0\nprotobuf==4.25.3\npydantic==2.7.4\nrich==13.7.1\nRequests==2.32.3\nsafetensors==0.4.5\nsoundfile==0.12.1\ntransformers==4.44.1\nuvicorn==0.32.0\nwget==3.2\nWeTextProcessing==1.0.3\ntorch==2.6.0\ntorchaudio==2.6.0\naccelerate==1.0.1\nlangchain==0.2.17\nlangchain-openai==0.1.25\nlangchain-huggingface==0.0.3\nlangchain-chroma==0.1.4\nrouge_score\njiwer==3.0.5\nipykernel\nfaster-whisper==1.1.0\nfairseq2==0.4.5\nsonar-space==0.4.0\nlangchain-core==0.2.43\nlangchain-text-splitters==0.2.4\nsentence-transformers==3.1.1\nchromadb==0.5.23\nonnxruntime==1.19.2\nctranslate2==4.5.0\neinops==0.8.0\nsentencepiece==0.2.0\n'
CONSTRAINTS = """torch==2.6.0+cu124
torchaudio==2.6.0+cu124
fairseq2==0.4.5
fairseq2n==0.4.5+cu124
sonar-space==0.4.0
numpy==1.24.4
"""


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


def move_qwen_imports(path):
    original = path.read_text(encoding="utf-8")
    tree = ast.parse(original)
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            if node.module == "qwen_omni_utils" and names == {"process_mm_info"}:
                top_imports.append(node)
            elif node.module == "transformers" and names == {
                "Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniProcessor"
            }:
                top_imports.append(node)
    if not top_imports:
        print("Qwen top-level imports absent; keeping existing source.", flush=True)
        return
    if len(top_imports) != 2:
        raise RuntimeError("Unexpected Qwen imports. Source was not changed; inspect model.py.")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen_Omni")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    insertions = {
        methods["__init__"].body[0].lineno: (
            "        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor\n"
        ),
        methods["generate"].body[0].lineno: (
            "        from qwen_omni_utils import process_mm_info\n"
        ),
    }
    removals = {i for n in top_imports for i in range(n.lineno, n.end_lineno + 1)}
    output = []
    for i, line in enumerate(original.splitlines(keepends=True), 1):
        if i in insertions:
            output.append(insertions[i])
        if i not in removals:
            output.append(line)
    revised = "".join(output)
    compile(revised, str(path), "exec")
    backup = path.with_name(path.name + ".before-qwen-lazy-import.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(revised, encoding="utf-8")
    print(f"Moved Qwen imports into Qwen methods; original backup: {backup}", flush=True)


def main():
    root = Path.cwd()
    model = root / "src/spoken_chatbot/model.py"
    if not model.is_file() or not (root / "requirements.txt").is_file():
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
    req = root / "requirements-server.txt"
    constraints = root / "constraints-server.txt"
    req.write_text(REQUIREMENTS, encoding="utf-8")
    constraints.write_text(CONSTRAINTS, encoding="utf-8")
    print(f"Installing remaining dependencies with {sys.executable}", flush=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "-r", str(req), "-c", str(constraints),
    ], check=True)
    move_qwen_imports(model)
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
