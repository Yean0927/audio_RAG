"""Local speech-tokenizer discovery for the GLM startup path."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from voice_web_demo import GLM_TOKENIZER_REPO, resolve_glm_tokenizer_path


class GLMTokenizerTests(unittest.TestCase):
    def setUp(self):
        workspace = self.enterContext(TemporaryDirectory())
        self.root = Path(workspace).resolve()
        self.cache = self.root / "hub"
        self.enterContext(patch("voice_service.glm.ROOT", self.root))
        self.enterContext(patch.dict(os.environ, {"TRANSFORMERS_CACHE": str(self.cache)}))
        self.enterContext(patch("requests.Session.request", side_effect=AssertionError("Unexpected network request")))
        self.model = self.root / "models" / "glm-4-voice-9b"
        self.model.mkdir(parents=True)

    def create_tokenizer(self, folder, *, is_pointer=False):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}")
        (folder / "preprocessor_config.json").write_text("{}")
        weights = b"version https://git-lfs.github.com/spec/v1\n" if is_pointer else bytes(1024)
        (folder / "model.safetensors").write_bytes(weights)

    def resolve(self, path=GLM_TOKENIZER_REPO):
        return resolve_glm_tokenizer_path(path, str(self.model))

    def test_explicit_tokenizer_path_takes_precedence(self):
        self.create_tokenizer(self.root / "glm-4-voice-tokenizer")
        self.assertEqual(self.resolve("/models/selected-voice-tokenizer"), "/models/selected-voice-tokenizer")

    def test_tokenizer_next_to_main_model_is_used_locally(self):
        tokenizer = self.model.parent / "glm-4-voice-tokenizer"
        self.create_tokenizer(tokenizer)
        self.assertEqual(self.resolve(), str(tokenizer))

    def test_incomplete_project_copy_falls_back_to_legacy_cache(self):
        self.create_tokenizer(self.root / "glm-4-voice-tokenizer", is_pointer=True)
        repo = self.cache / "models--THUDM--glm-4-voice-tokenizer"
        revision = "a" * 40
        tokenizer = repo / "snapshots" / revision
        self.create_tokenizer(tokenizer)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(revision)
        self.assertEqual(self.resolve(), str(tokenizer))

    def test_legacy_project_tokenizer_is_used_locally(self):
        tokenizer = self.root / "THUDM" / "glm-4-voice-tokenizer"
        self.create_tokenizer(tokenizer)
        self.assertEqual(self.resolve(), str(tokenizer))

    def test_missing_local_tokenizer_keeps_hub_fallback(self):
        self.assertEqual(self.resolve(), GLM_TOKENIZER_REPO)
