import pytest

from wandbot.models.llm import AsyncGoogleGenAILLMModel


class _DummyClient:
    class _Aio:
        class _Models:
            async def generate_content(self, *args, **kwargs):
                raise RuntimeError("not used in unit tests")

        models = _Models()

    aio = _Aio()


@pytest.fixture
def make_google_model(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(
        "wandbot.models.llm.genai.Client", lambda api_key: _DummyClient()
    )

    def _factory(**kwargs):
        return AsyncGoogleGenAILLMModel(model_name="gemini-3-flash-preview", **kwargs)

    return _factory


def test_coerce_thinking_budget():
    assert AsyncGoogleGenAILLMModel._coerce_thinking_budget(500) == 500
    assert AsyncGoogleGenAILLMModel._coerce_thinking_budget("500") == 500
    assert AsyncGoogleGenAILLMModel._coerce_thinking_budget(500.9) == 500
    assert AsyncGoogleGenAILLMModel._coerce_thinking_budget(0) is None
    assert AsyncGoogleGenAILLMModel._coerce_thinking_budget("0") is None
    assert AsyncGoogleGenAILLMModel._coerce_thinking_budget("abc") is None


def test_model_supports_thinking_controls():
    assert AsyncGoogleGenAILLMModel._model_supports_thinking_controls(
        "gemini-3-flash-preview"
    )
    assert not AsyncGoogleGenAILLMModel._model_supports_thinking_controls(
        "gemini-2.5-flash"
    )
    assert not AsyncGoogleGenAILLMModel._model_supports_thinking_controls("gpt-4o")


def test_build_thinking_config_uses_budget_for_gemini3(make_google_model):
    model = make_google_model(thinking_budget=500)
    thinking_config = model._build_thinking_config()

    assert thinking_config is not None
    assert thinking_config.thinking_budget == 500


def test_build_thinking_config_uses_level_for_gemini3(make_google_model):
    model = make_google_model(thinking_level="minimal")
    thinking_config = model._build_thinking_config()

    assert thinking_config is not None
    assert str(thinking_config.thinking_level).upper().endswith("MINIMAL")


def test_build_thinking_config_ignores_zero_budget(make_google_model):
    model = make_google_model(thinking_budget=0)
    thinking_config = model._build_thinking_config()

    assert thinking_config is None


def test_build_thinking_config_rejects_level_and_budget_together(
    make_google_model,
):
    model = make_google_model(thinking_level="minimal", thinking_budget=256)
    with pytest.raises(
        ValueError,
        match="Set either thinking_level or thinking_budget for Gemini 3, not both.",
    ):
        model._build_thinking_config()
