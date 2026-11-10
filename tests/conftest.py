import pytest

from engram import Brain, Settings


@pytest.fixture
def brain(tmp_path):
    """A brain that costs nothing: fake LLM + hash embedder + temp database."""
    s = Settings(db_path=str(tmp_path / "test.db"), llm="fake", embedder="hash",
                 cache_dir=str(tmp_path / "cache"), use_cache=False, api_keys="")
    return Brain(s)
