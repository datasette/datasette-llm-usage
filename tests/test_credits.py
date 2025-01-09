import pytest
from datasette.app import Datasette
import time
from sqlite_utils.db import Database
from datasette_llm_usage.db import migration
from datasette_llm_usage import subtract_credits, TokensExhausted


class DbWrapper:
    def __init__(self, sqlite_utils_db, datasette_db):
        self.db = sqlite_utils_db
        self.datasette_db = datasette_db


_test_db_suffix = 1


@pytest.fixture
def db():
    global _test_db_suffix
    memory_name = "test_db_{}".format(_test_db_suffix)
    _test_db_suffix += 1
    db = Database(memory=True, memory_name=memory_name)
    # Apply migrations
    migration.apply(db)
    # Set up test data
    db["_llm_usage_provider"].insert({"id": 1, "name": "test-provider"})
    db["_llm_usage_model"].insert(
        {
            "id": 1,
            "name": "test-model",
            "provider_id": 1,
            "tiers": '[{"max_tokens": null, "input_cost": 10, "output_cost": 20}]',
        }
    )
    ds = Datasette()
    ds.add_memory_database(memory_name)
    return DbWrapper(db, ds.get_database(memory_name))


@pytest.fixture
def mock_time(monkeypatch):
    current_time = [time.time()]

    def mock_time_fn():
        return current_time[0]

    monkeypatch.setattr(time, "time", mock_time_fn)
    return current_time


@pytest.mark.asyncio
async def test_basic_credit_subtraction(db):
    # Set up initial allowance
    db.db["_llm_usage_allowance"].insert(
        {
            "provider_id": 1,
            "credits_remaining": 1000,
            "last_reset_timestamp": int(time.time()),
            "daily_reset_amount": 1000,
        }
    )

    # Subtract some credits (10 input tokens, 5 output tokens)
    await subtract_credits(db.datasette_db, "test-model", 10, 5)

    # Check remaining credits (10*10 + 5*20 = 200 credits used)
    remaining = db.db["_llm_usage_allowance"].get(1)["credits_remaining"]
    assert remaining == 800


@pytest.mark.asyncio
async def test_insufficient_credits(db):
    # Set up low initial allowance
    db.db["_llm_usage_allowance"].insert(
        {
            "provider_id": 1,
            "credits_remaining": 100,
            "last_reset_timestamp": int(time.time()),
            "daily_reset_amount": 1000,
        }
    )

    # Try to subtract more credits than available
    with pytest.raises(TokensExhausted):
        await subtract_credits(db.datasette_db, "test-model", 100, 100)

    # Verify credits weren't changed
    remaining = db.db["_llm_usage_allowance"].get(1)["credits_remaining"]
    assert remaining == 100


@pytest.mark.asyncio
async def test_daily_reset(db, mock_time):
    initial_time = int(mock_time[0])

    # Set up depleted allowance
    db.db["_llm_usage_allowance"].insert(
        {
            "provider_id": 1,
            "credits_remaining": 100,
            "last_reset_timestamp": initial_time,
            "daily_reset_amount": 1000,
        }
    )

    # Advance time by 25 hours
    mock_time[0] = initial_time + (25 * 3600)

    # Try to subtract some credits
    await subtract_credits(db.datasette_db, "test-model", 10, 5)

    # Should have reset to 1000 first, then subtracted
    row = db.db["_llm_usage_allowance"].get(1)
    assert row["credits_remaining"] == 800  # 1000 - 200
    assert row["last_reset_timestamp"] == initial_time + (25 * 3600)


@pytest.mark.asyncio
async def test_no_daily_reset_if_too_soon(db, mock_time):
    initial_time = int(mock_time[0])

    # Set up allowance
    db.db["_llm_usage_allowance"].insert(
        {
            "provider_id": 1,
            "credits_remaining": 500,
            "last_reset_timestamp": initial_time,
            "daily_reset_amount": 1000,
        }
    )

    # Advance time by 23 hours
    mock_time[0] = initial_time + (23 * 3600)

    # Subtract some credits
    await subtract_credits(db.datasette_db, "test-model", 10, 5)

    row = db.db["_llm_usage_allowance"].get(1)
    assert row["credits_remaining"] == 300  # 500 - 200
    assert row["last_reset_timestamp"] == initial_time  # Shouldn't have changed


@pytest.mark.asyncio
async def test_daily_reset_disabled(db, mock_time):
    initial_time = int(mock_time[0])

    # Set up allowance with no daily reset
    db.db["_llm_usage_allowance"].insert(
        {
            "provider_id": 1,
            "credits_remaining": 500,
            "last_reset_timestamp": initial_time,
            "daily_reset_amount": None,  # Disabled
        }
    )

    # Advance time by 48 hours
    mock_time[0] = initial_time + (48 * 3600)

    # Subtract some credits
    await subtract_credits(db.datasette_db, "test-model", 10, 5)

    row = db.db["_llm_usage_allowance"].get(1)
    assert row["credits_remaining"] == 300  # Should subtract without reset
    assert row["last_reset_timestamp"] == initial_time
