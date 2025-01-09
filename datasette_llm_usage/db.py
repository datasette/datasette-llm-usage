import llm
from sqlite_utils import Database
from sqlite_migrate import Migrations
import sys
import json
import time


migration = Migrations("datasette-llm-usage")


@migration()
def create_tables(db):
    db["_llm_usage_provider"].create(
        {
            "id": int,
            "name": str,
        },
        pk="id",
    ).create_index(["name"], unique=True)
    db["_llm_usage_model"].create(
        {
            "id": int,
            "name": str,
            "provider_id": int,
            "tiers": str,
        },
        pk="id",
        foreign_keys=(("provider_id", "_llm_usage_provider", "id"),),
    ).create_index(["name"], unique=True)
    db["_llm_usage"].create(
        {
            "id": int,
            "created": int,
            "provider_id": int,
            "model_id": int,
            "purpose": str,
            "actor_id": str,
            "input_tokens": int,
            "output_tokens": int,
            "credits": int,
        },
        pk="id",
        foreign_keys=(
            ("provider_id", "_llm_usage_provider", "id"),
            ("model_id", "_llm_usage_model", "id"),
        ),
    )
    db["_llm_usage_allowance"].create(
        {
            "provider_id": int,
            "credits_remaining": int,
            "last_reset_timestamp": int,
            "daily_reset_amount": int,  # null = no reset
        },
        pk="provider_id",
        not_null=("provider_id", "credits_remaining"),
        foreign_keys=(("provider_id", "_llm_usage_provider", "id"),),
    )


def make_populate(datasette):
    def inner(conn):
        db = Database(conn)
        # Create or update records for each model
        plugin_config = datasette.plugin_config("datasette-llm-usage") or {}
        models = plugin_config.get("models") or []
        allowances = plugin_config.get("allowances") or []
        if not models:
            return
        for model in models:
            # provider: gemini
            # model_id: gemini-1.5-pro
            # tiers:
            # - max_tokens: 128000
            #     input_cost: 125
            #     output_cost: 500
            # - max_tokens: null
            #     input_cost: 250
            #     output_cost: 1000
            provider = model["provider"]
            alias = model["model_id"]
            try:
                model_id = llm.get_async_model(alias).model_id
            except llm.UnknownModelError:
                print("Unknown model alias: {}".format(alias), file=sys.stderr)
                continue
            tiers = model["tiers"]
            provider_id = db["_llm_usage_provider"].lookup({"name": provider})
            sql = """
            INSERT INTO _llm_usage_model (name, provider_id, tiers)
            VALUES (:name, :provider_id, :tiers)
            ON CONFLICT (name) DO UPDATE SET
                provider_id = excluded.provider_id,
                tiers = excluded.tiers
            """
            db.execute(
                sql,
                {
                    "name": model_id,
                    "provider_id": provider_id,
                    "tiers": json.dumps(tiers),
                },
            )
        for allowance in allowances:
            # - provider: gemini
            #   daily_reset_amount: 100000
            provider_id = db["_llm_usage_provider"].lookup(
                {"name": allowance["provider"]}
            )
            daily_reset_amount = allowance["daily_reset_amount"]
            sql = """
            INSERT INTO _llm_usage_allowance (provider_id, credits_remaining, last_reset_timestamp, daily_reset_amount)
            VALUES (:provider_id, :credits_remaining, :last_reset_timestamp, :daily_reset_amount)
            ON CONFLICT (provider_id) DO UPDATE SET
                credits_remaining = excluded.credits_remaining,
                last_reset_timestamp = excluded.last_reset_timestamp,
                daily_reset_amount = excluded.daily_reset_amount
            """
            params = {
                "provider_id": provider_id,
                "credits_remaining": daily_reset_amount,
                "last_reset_timestamp": int(time.time()),
                "daily_reset_amount": daily_reset_amount,
            }
            db.execute(sql, params)

    return inner


async def get_database(datasette, migrate=False, populate=False):
    plugin_config = datasette.plugin_config("datasette-llm-usage") or {}
    db_name = plugin_config.get("database")
    if db_name:
        db = datasette.get_database(db_name)
    else:
        db = datasette.get_internal_database()
    if migrate:
        await db.execute_write_fn(lambda conn: migration.apply(Database(conn)))
    if populate:
        await db.execute_write_fn(make_populate(datasette))
    return db
