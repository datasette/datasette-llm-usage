from __future__ import annotations
from dataclasses import dataclass
from datasette import hookimpl, Response
import json
from sqlite_migrate import Migrations
from sqlite_utils import Database
import time
from typing import Optional
import llm
from .pricing import calculate_costs


@dataclass
class ModelAllowance:
    model_id: str
    provider: str
    credits_remaining: int
    tokens_remaining: int


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
            "purpose": str,
            "credits_remaining": int,
            "last_reset_timestamp": int,
            "daily_reset_amount": int,  # null = no reset
        },
        pk=("provider_id", "purpose"),
        not_null=("provider_id", "purpose", "credits_remaining"),
        foreign_keys=(("provider_id", "_llm_usage_provider", "id"),),
    )


async def llm_usage_simple_prompt(datasette, request):
    if not request.actor:
        return Response.text("Not logged in", status=403)
    llm = LLM(datasette)
    post_data = await request.post_vars()
    prompt = post_data.get("prompt")
    allowances = await llm.get_model_allowances()
    context = {
        "allowances": allowances,
    }

    if prompt:
        model_id = post_data.get("model") or "gpt-4o-mini"
        model = llm.get_async_model(model_id, purpose="simple_prompt")
        response = await model.prompt(prompt, actor_id=request.actor["id"])
        text = await response.text()
        usage = await response.usage()
        context.update(
            {
                "prompt": prompt,
                "model_id": model_id,
                "prompt_response": text,
                "input_tokens": usage.input,
                "output_tokens": usage.output,
            }
        )

    return Response.html(
        await datasette.render_template(
            "llm_usage_single_prompt.html", context, request=request
        )
    )


@hookimpl
def register_routes():
    return [
        (r"^/-/llm-usage-simple-prompt$", llm_usage_simple_prompt),
    ]


@hookimpl
def startup(datasette):
    async def inner():
        await get_database(datasette, migrate=True, populate=True)

    return inner


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
            model_id = model["model_id"]
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
            purpose = allowance.get("purpose") or ""
            daily_reset_amount = allowance["daily_reset_amount"]
            sql = """
            INSERT INTO _llm_usage_allowance (provider_id, purpose, credits_remaining, last_reset_timestamp, daily_reset_amount)
            VALUES (:provider_id, :purpose, :credits_remaining, :last_reset_timestamp, :daily_reset_amount)
            ON CONFLICT (provider_id, purpose) DO UPDATE SET
                credits_remaining = excluded.credits_remaining,
                last_reset_timestamp = excluded.last_reset_timestamp,
                daily_reset_amount = excluded.daily_reset_amount
            """
            params = {
                "provider_id": provider_id,
                "purpose": purpose,
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


async def subtract_credits(db, purpose, model_id, input_tokens, output_tokens):
    # Calculate cost using the new pricing module
    input_cost, output_cost = await calculate_costs(
        db, model_id, input_tokens, output_tokens
    )
    cost = int((input_cost + output_cost) * 1_000_000)  # Convert to credits

    # Lookup the model_db_id and provider_db_id for this model_id
    row = (
        await db.execute(
            "select id, provider_id from _llm_usage_model where name = :name",
            {"name": model_id},
        )
    ).first()
    if not row:
        raise ValueError("Unknown model_id: {}".format(model_id))
    provider_db_id = row["provider_id"]
    model_db_id = row["id"]
    row_purpose = ""
    if purpose:
        # Is there a purpose row?
        sql = "select * from _llm_usage_allowance where provider_id = :provider_id and purpose = :purpose"
        if (
            await db.execute(sql, {"provider_id": provider_db_id, "purpose": purpose})
        ).first():
            row_purpose = purpose
    await db.execute_write(
        """
        update _llm_usage_allowance
        set credits_remaining = credits_remaining - :cost
        where provider_id = :provider_id and purpose = :purpose
        """,
        {
            "provider_id": provider_db_id,
            "purpose": row_purpose,
            "cost": cost,
        },
    )


class WrappedModel:
    def __init__(self, model: llm.AsyncModel, datasette, purpose: Optional[str] = None):
        self.model = model
        self.datasette = datasette
        self.purpose = purpose

    async def prompt(
        self,
        prompt: Optional[str],
        system: Optional[str] = None,
        actor_id: Optional[str] = None,
        **kwargs,
    ):
        response = self.model.prompt(prompt, system=system, **kwargs)

        async def done(response):
            # Log usage against current actor_id and purpose
            usage = await response.usage()
            input_tokens = usage.input
            output_tokens = usage.output
            db = await get_database(self.datasette)
            row = (
                await db.execute(
                    "select id, provider_id from _llm_usage_model where name = :name",
                    {"name": self.model.model_id},
                )
            ).first()
            provider_id = row["provider_id"]
            model_id = row["id"]

            # Calculate credits using the new pricing module
            input_cost, output_cost = await calculate_costs(
                db, self.model.model_id, input_tokens, output_tokens
            )
            credits = int((input_cost + output_cost) * 1_000_000)  # Convert to credits

            await db.execute_write(
                """
            insert into _llm_usage (created, provider_id, model_id, purpose, actor_id, input_tokens, output_tokens, credits)
            values (:created, :provider_id, :model_id, :purpose, {actor_id}, :input_tokens, :output_tokens, :credits)
            """.format(
                    actor_id=":actor_id" if actor_id else "null",
                ),
                {
                    "created": int(time.time() * 1000),
                    "provider_id": provider_id,
                    "model_id": model_id,
                    "purpose": self.purpose,
                    "actor_id": actor_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "credits": credits,
                },
            )
            # Subtract the appropriate amount of credits from the allowance
            await subtract_credits(
                db, self.purpose, self.model.model_id, input_tokens, output_tokens
            )

        await response.on_done(done)
        return response

    def __repr__(self):
        return f"WrappedModel: {self.model.model_id}"


class LLM:
    def __init__(self, datasette):
        self.datasette = datasette

    async def get_async_models(self):
        models = [
            WrappedModel(model, self.datasette) for model in llm.get_async_models()
        ]
        # Filter for the ones in the database
        db = await get_database(self.datasette)
        model_ids = [
            row["name"] for row in await db.execute("select name from _llm_usage_model")
        ]
        return [model for model in models if model.model.model_id in model_ids]

    def get_async_model(self, model_id=None, purpose=None):
        return WrappedModel(
            llm.get_async_model(model_id), self.datasette, purpose=purpose
        )

    async def get_model_allowances(self):
        # Returns list of ModelAllowance objects
        # class ModelAllowance:
        #     model_id: - the name from _llm_usage_model
        #     provider: - the name from _llm_usage_provider
        #     credits_remaining: int - from _llm_usage_allowance
        #     tokens_remaining: int - calculated based on lowest pricing tier
        db = await get_database(self.datasette)
        rows = [
            dict(row)
            for row in (
                await db.execute(
                    """
            select
                _llm_usage_model.name as model_id,
                _llm_usage_provider.name as provider,
                _llm_usage_allowance.credits_remaining,
                _llm_usage_model.tiers
            from _llm_usage_model
            join _llm_usage_provider on _llm_usage_model.provider_id = _llm_usage_provider.id
            join _llm_usage_allowance on _llm_usage_provider.id = _llm_usage_allowance.provider_id
            """
                )
            ).rows
        ]
        # Just consider input tokens price of first tier for the moment
        available = []
        for row in rows:
            tiers = json.loads(row.pop("tiers"))
            input_cost = tiers[0]["input_cost"]
            row["tokens_remaining"] = row["credits_remaining"] // input_cost
            # Is the model plugin installed?
            try:
                llm.get_async_model(row["model_id"])
                available.append(ModelAllowance(**row))
            except llm.UnknownModelError:
                pass
        return available

    async def has_allowance(self, purpose: Optional[str] = None):
        db = await get_database(self.datasette)
        if purpose:
            #  First check allowance for this purpose
            sql = "select credits_remaining from _llm_usage_allowance where purpose = :purpose"
            credits_remaining = (
                await db.execute(sql, {"purpose": purpose})
            ).single_value()
            if credits_remaining > 0:
                return True
        # Check general allowance instead
        credits_remaining = await db.execute(
            "select credits_remaining from _llm_usage_allowance where purpose = ''"
        ).single_value()
        return credits_remaining > 0
