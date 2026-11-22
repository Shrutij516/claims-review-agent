import pytest

from claims_review.llm import DEFAULT_MODEL, GroqLLM, MissingAPIKeyError


def test_missing_key_raises_a_clear_error_that_names_the_variable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError) as err:
        GroqLLM()
    assert "GROQ_API_KEY" in str(err.value)


def test_blank_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    with pytest.raises(MissingAPIKeyError):
        GroqLLM()


def test_key_value_never_appears_in_repr_or_error(monkeypatch):
    secret = "gsk-test-NOT-A-REAL-KEY-12345"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    llm = GroqLLM()          # builds a client; makes no network call
    assert secret not in repr(llm) and secret not in str(vars(llm).get("model"))
    monkeypatch.delenv("GROQ_API_KEY")
    with pytest.raises(MissingAPIKeyError) as err:
        GroqLLM()
    assert secret not in str(err.value)


def test_model_comes_from_env_with_a_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-NOT-A-REAL-KEY-12345")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    assert GroqLLM().model == DEFAULT_MODEL == "openai/gpt-oss-120b"
    monkeypatch.setenv("GROQ_MODEL", "some/other-model")
    assert GroqLLM().model == "some/other-model"


class _StubClient:
    def __init__(self, content):
        self.calls = []
        outer = self

        class Completions:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                message = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

        self.chat = type("Chat", (), {"completions": Completions()})()


def test_complete_uses_temperature_zero_model_timeout_and_returns_text():
    client = _StubClient("hello")
    llm = GroqLLM(model="m", client=client)
    assert llm.complete([{"role": "user", "content": "hi"}]) == "hello"
    call = client.calls[0]
    assert call["temperature"] == 0 and call["model"] == "m" and call["timeout"] == 60.0
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_complete_returns_empty_string_when_the_model_returns_no_content():
    assert GroqLLM(model="m", client=_StubClient(None)).complete([]) == ""
