import pytest

from wandbot.rag.response_synthesis import ResponseSynthesizer
from wandbot.schema.document import Document
from wandbot.schema.retrieval import RetrievalResult


async def fake_stream(*args, **kwargs):
    for tok in ["hello ", "world"]:
        yield tok


@pytest.mark.asyncio
async def test_streaming_response():
    synth = ResponseSynthesizer(
        primary_provider="openai",
        primary_model_name="dummy",
        primary_temperature=0,
        fallback_provider="openai",
        fallback_model_name="dummy",
        fallback_temperature=0,
        max_retries=1,
    )

    synth.model.stream = fake_stream  # type: ignore
    synth.get_messages = lambda x: []

    retrieval = RetrievalResult(
        documents=[Document(page_content="doc", metadata={"source": "s"})],
        retrieval_info={"query": "q", "language": "en", "intents": [], "sub_queries": []},
    )

    tokens = []
    async for token in synth.stream(retrieval):
        tokens.append(token)

    assert "".join(tokens) == "hello world"
    assert synth.stream_output["response"] == "hello world"
