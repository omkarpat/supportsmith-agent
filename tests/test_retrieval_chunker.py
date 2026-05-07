from app.retrieval.chunker import FAQChunker


def test_faq_chunker_returns_one_chunk_per_row() -> None:
    chunker = FAQChunker()

    chunks = chunker.split(
        title="Reset password",
        body="Use settings -> change password.",
    )

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.title == "Reset password"
    assert chunk.content == "Q: Reset password\n\nA: Use settings -> change password."
    assert chunk.embedding_text == "reset password use settings -> change password."


def test_faq_chunker_strips_outer_whitespace() -> None:
    chunker = FAQChunker()

    chunks = chunker.split(title="  Reset password  ", body="  Use settings.\n")

    assert chunks[0].title == "Reset password"
    assert "Q: Reset password" in chunks[0].content
    assert chunks[0].embedding_text == "reset password use settings."
