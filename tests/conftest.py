"""Shared test helpers: a scripted fake LLM. No test makes a network call."""


class FakeLLM:
    """Returns scripted replies in order and records every call.

    A reply that is an Exception instance is raised instead of returned. A call with no reply left
    fails the test, so an unexpected extra LLM call cannot go unnoticed.
    """
    model = "fake-llm"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError(f"unexpected LLM call number {len(self.calls)}")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
