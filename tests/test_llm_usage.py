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
    assert set(table_names).issuperset(
        (
            "_llm_usage_provider",
            "_llm_usage_model",
            "_llm_usage",
            "_llm_usage_allowance",
        )
    )


@pytest.mark.asyncio
async def test_counts_usage(tmpdir):
    internal = tmpdir / "internal.db"
    datasette = Datasette(internal=str(internal))
    await datasette.invoke_startup()
    # Set up an allowance
    db = datasette.get_internal_database()

    # Set up provider and model first
    await db.execute_write(
        """
        insert into _llm_usage_provider (id, name) values (1, 'mock')
        """
    )
    await db.execute_write(
        """
        insert into _llm_usage_model (id, name, provider_id, tiers) 
        values (1, 'async-mock', 1, '[{"max_tokens": null, "input_cost": 100, "output_cost": 100}]')
        """
    )
    await db.execute_write(
        """
        insert into _llm_usage_allowance (provider_id, credits_remaining, daily_reset_amount, last_reset_timestamp)
        values (1, 10000, 10000, strftime('%s', 'now'))
        """
    )

    llm = LLM(datasette)
    models = await llm.get_async_models()
    model_ids = [m.model.model_id for m in models]
    assert "async-mock" in model_ids
    model = await llm.get_async_model("async-mock")
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
    assert row["model_id"] == 1
    assert row["provider_id"] == 1
    assert row["input_tokens"] == 1
    assert row["output_tokens"] == 1
    assert row["purpose"] is None

    allowance_rows = (await db.execute("select * from _llm_usage_allowance")).rows
    allowance_row = dict(allowance_rows[0])
    assert allowance_row["credits_remaining"] == 9800  # 10000 - (1*100 + 1*100)
