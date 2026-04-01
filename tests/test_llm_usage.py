from datasette.app import Datasette
from datasette import hookimpl
from datasette.permissions import PermissionSQL
from datasette.plugins import pm
from datasette_llm import LLM
import json
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


@pytest.mark.asyncio
async def test_counts_usage_for_chain_responses(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    db = datasette.get_internal_database()
    llm = LLM(datasette)
    model = await llm.model("echo", actor={"id": "test-user"})

    def example(input: str) -> str:
        return f"Example output for {input}"

    chain_response = model.chain(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "name": "example",
                        "arguments": {"input": "test"},
                    }
                ],
                "prompt": "hello world",
            }
        ),
        tools=[example],
    )
    responses = []
    async for response in chain_response.responses():
        responses.append(await response.text())

    assert len(responses) == 2

    usage_rows = (await db.execute("select * from llm_usage order by rowid")).rows
    assert len(usage_rows) == 2
    assert all(dict(row)["model"] == "echo" for row in usage_rows)
    assert all(dict(row)["actor_id"] == "test-user" for row in usage_rows)


@pytest.mark.asyncio
async def test_tool_calls_logged(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(
        internal=str(internal),
        metadata={"plugins": {"datasette-llm-usage": {"log_prompts": True}}},
    )
    await datasette.invoke_startup()
    db = datasette.get_internal_database()
    llm = LLM(datasette)
    model = await llm.model("echo", actor={"id": "test-user"})

    def example(input: str) -> str:
        return f"Example output for {input}"

    chain_response = model.chain(
        json.dumps(
            {
                "tool_calls": [
                    {
                        "name": "example",
                        "arguments": {"input": "test"},
                    }
                ],
                "prompt": "hello world",
            }
        ),
        tools=[example],
    )
    responses = []
    async for response in chain_response.responses():
        responses.append(await response.text())

    assert len(responses) == 2

    log_rows = (
        await db.execute("select * from llm_usage_prompt_log order by rowid")
    ).rows
    assert len(log_rows) == 2

    # First response: model made tool calls, no tool results input
    first = dict(log_rows[0])
    assert first["tool_results"] is None
    tool_calls = json.loads(first["tool_calls"])
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "example"
    assert tool_calls[0]["arguments"] == {"input": "test"}

    # Second response: received tool results, no further tool calls
    second = dict(log_rows[1])
    assert second["tool_calls"] is None
    tool_results = json.loads(second["tool_results"])
    assert len(tool_results) == 1
    assert tool_results[0]["name"] == "example"
    assert "Example output for test" in tool_results[0]["output"]


@pytest.mark.asyncio
async def test_simple_prompt_permission_denied(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    # No actor at all
    response = await datasette.client.get("/-/llm-usage-simple-prompt")
    assert response.status_code == 403
    # Actor without permission
    response = await datasette.client.get(
        "/-/llm-usage-simple-prompt",
        cookies={"ds_actor": datasette.sign({"a": {"id": "user1"}}, "actor")},
    )
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_simple_prompt_permission_allowed(tmpdir):
    internal = tmpdir / "internal.db"

    class AllowPromptPlugin:
        __name__ = "AllowPromptPlugin"

        @hookimpl
        def permission_resources_sql(self, datasette, actor, action):
            if action == "llm-usage-simple-prompt":
                return PermissionSQL.allow(reason="test")

    plugin = AllowPromptPlugin()
    pm.register(plugin)
    try:
        datasette = Datasette(internal=str(internal))
        await datasette.invoke_startup()
        response = await datasette.client.get(
            "/-/llm-usage-simple-prompt",
            cookies={"ds_actor": datasette.sign({"a": {"id": "user1"}}, "actor")},
        )
        assert response.status_code == 200
        assert "<h1>Simple prompt</h1>" in response.text
        assert "<select" in response.text
    finally:
        pm.unregister(plugin)
