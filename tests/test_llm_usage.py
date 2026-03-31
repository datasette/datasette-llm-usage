from datasette.app import Datasette
from datasette_llm import LLM
import pytest


@pytest.mark.asyncio
async def test_tables_created(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    db = datasette.get_internal_database()
    table_names = await db.table_names()
    assert "llm_usage" in table_names
    assert "llm_usage_prompt_log" in table_names


@pytest.mark.asyncio
async def test_counts_usage(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    db = datasette.get_internal_database()
    llm = LLM(datasette)
    model = await llm.model("echo", actor={"id": "test-user"})
    response = await model.prompt("hello world")
    text = await response.text()
    assert "hello world" in text
    usage = await response.usage()
    assert usage.input > 0
    assert usage.output > 0
    # Verify usage was recorded
    usage_rows = (await db.execute("select * from llm_usage")).rows
    assert len(usage_rows) == 1
    row = dict(usage_rows[0])
    assert row["model"] == "echo"
    assert row["input_tokens"] == usage.input
    assert row["output_tokens"] == usage.output
    assert row["actor_id"] == "test-user"
    # Prompt log should not be populated by default
    log_rows = (await db.execute("select * from llm_usage_prompt_log")).rows
    assert len(log_rows) == 0


@pytest.mark.asyncio
async def test_prompt_logging(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(
        internal=str(internal),
        metadata={"plugins": {"datasette-llm-usage": {"log_prompts": True}}},
    )
    await datasette.invoke_startup()
    db = datasette.get_internal_database()
    llm = LLM(datasette)
    model = await llm.model("echo", actor={"id": "test-user"})
    response = await model.prompt("hello world")
    text = await response.text()
    assert "hello world" in text
    # Verify prompt was logged
    log_rows = (await db.execute("select * from llm_usage_prompt_log")).rows
    assert len(log_rows) == 1
    row = dict(log_rows[0])
    assert row["prompt"] == "hello world"
    assert row["response"] == text
    assert row["model"] == "echo"
    assert row["actor_id"] == "test-user"


@pytest.mark.asyncio
async def test_no_actor(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    db = datasette.get_internal_database()
    llm = LLM(datasette)
    model = await llm.model("echo")
    response = await model.prompt("hello")
    await response.text()
    usage_rows = (await db.execute("select * from llm_usage")).rows
    assert len(usage_rows) == 1
    row = dict(usage_rows[0])
    assert row["actor_id"] is None
