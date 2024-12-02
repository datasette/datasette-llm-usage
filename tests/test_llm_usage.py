from datasette.app import Datasette
from datasette_llm_usage import LLM
import pytest


@pytest.mark.asyncio
async def test_tables_created(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    db = datasette.get_internal_database()
    table_names = await db.table_names()
    assert "_llm_allowance" in table_names
    assert "_llm_usage" in table_names


@pytest.mark.asyncio
async def test_counts_usage(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    # Set up an allowance
    db = datasette.get_internal_database()
    await db.execute_write(
        """
        insert into _llm_allowance (id, created, credits_remaining, daily_reset, daily_reset_amount) values (1, 0, 10000, 0, 0)
    """
    )
    llm = LLM(datasette)
    models = llm.get_async_models()
    model_ids = [m.model.model_id for m in models]
    assert "gpt-4o-mini" in model_ids
    assert "async-mock" in model_ids
    model = llm.get_async_model("async-mock")
    model.model.enqueue(["hello there"])
    response = await model.prompt("hello")
    usage = await response.usage()
    text = await response.text()
    assert text == "hello there"
    assert usage.input == 1
    assert usage.output == 1
    # It should be written to the table
    usage_rows = (await db.execute("select * from _llm_usage")).rows
    row = dict(usage_rows[0])
    assert row["model"] == "async-mock"
    assert row["input_tokens"] == 1
    assert row["output_tokens"] == 1
    assert row["purpose"] is None
    allowance_rows = (await db.execute("select * from _llm_allowance")).rows
    allowance_row = dict(allowance_rows[0])
    assert allowance_row["credits_remaining"] == 8900
