"""Lightweight tests for config dataclasses and env parsing."""

from free_gpu_llm.config import GenerationConfig, ModelConfig, get_hf_token


def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.max_new_tokens == 256
    assert cfg.do_sample is True


def test_model_config_requires_name_and_repo_id():
    cfg = ModelConfig(name="Gemma 3 4B", repo_id="google/gemma-3-4b-it")
    assert cfg.dtype == "bfloat16"
    assert cfg.quantization is None
    assert cfg.context_length == 2048


def test_model_config_nf4_variant():
    cfg = ModelConfig(name="Gemma 3 4B NF4", repo_id="google/gemma-3-4b-it", dtype="nf4", quantization="NF4")
    assert cfg.dtype == "nf4"
    assert cfg.quantization == "NF4"


def test_get_hf_token_reads_env_var(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_for_test")
    assert get_hf_token() == "hf_fake_token_for_test"


def test_get_hf_token_none_when_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert get_hf_token() is None
