from __future__ import annotations
from dataclasses import dataclass
from datasette import hookimpl, Response
from sqlite_migrate import Migrations
from sqlite_utils import Database
import time
from typing import Optional
import llm


@dataclass
class Price:
    name: str
    model_id: str
    size_limit: Optional[int]
    input_token_cost_10000th_cent: int
    output_token_cost_10000th_cent: int

    def cost_in_cents(self, input_tokens: int, output_tokens: int):
        return self.cost_in_credits(input_tokens, output_tokens) / 1000000

    def cost_in_credits(self, input_tokens: int, output_tokens: int):
        return (
            input_tokens * self.input_token_cost_10000th_cent
            + output_tokens * self.output_token_cost_10000th_cent
        )


PRICES = [
    Price("gemini-1.5-flash", "gemini-1.5-flash", 128000, 7, 30),
    Price("gemini-1.5-flash-128k", "gemini-1.5-flash", None, 15, 60),
    Price("gemini-1.5-flash-8b", "gemini-1.5-flash-8b", 128000, 3, 15),
    Price("gemini-1.5-flash-8b-128k", "gemini-1.5-flash-8b", None, 7, 30),
    Price("gemini-1.5-pro", "gemini-1.5-pro", 128000, 125, 500),
    Price("gemini-1.5-pro-128k", "gemini-1.5-pro", None, 250, 1000),
    Price("claude-3.5-sonnet", "claude-3.5-sonnet", None, 300, 1500),
    Price("claude-3-opus", "claude-3-opus", None, 1500, 7500),
    Price("claude-3-haiku", "claude-3-haiku", None, 25, 125),
    Price("claude-3.5-haiku", "claude-3.5-haiku", None, 100, 500),
    Price("gpt-4o", "gpt-4o", None, 250, 1000),
    Price("gpt-4o-mini", "gpt-4o-mini", None, 15, 60),
    Price("o1-preview", "o1-preview", None, 1500, 6000),
    Price("o1-mini", "o1-mini", None, 300, 1200),
    # This is just used by tests:
    Price("async-mock", "async-mock", 5, 100, 1000),
    Price("async-mock_gt_5", "async-mock", None, 120, 1200),
]


migration = Migrations("datasette_llm_usage")


@migration()
def create_usage_table(db):
    db["_llm_usage"].create(
        {
            "id": int,
            "created": int,
            "model": str,
            "purpose": str,
            "actor_id": str,
            "input_tokens": int,
            "output_tokens": int,
        },
        pk="id",
    )


@migration()
def create_allowance_table(db):
    db["_llm_allowance"].create(
        {
            "id": int,
            "created": int,
            "credits_remaining": int,
            "daily_reset": bool,
            "daily_reset_amount": int,
            "purpose": str,  # optional
        },
        pk="id",
        not_null=("id", "created", "credits_remaining"),
    )


async def llm_usage_simple_prompt(datasette, request):
    if not request.actor:
        return Response.text("Not logged in", status=403)
    llm = LLM(datasette)
    prompt = request.args.get("prompt")
    if not prompt:
        return Response.html("<form><input name=prompt><button>Submit</button></form>")
    model = llm.get_async_model("gpt-4o-mini", purpose="simple_prompt")
    response = await model.prompt(
        request.args.get("prompt"), actor_id=request.actor["id"]
    )
    text = await response.text()
    usage = await response.usage()
    return Response.json(
        {
            "text": text,
            "input_tokens": usage.input,
            "output_tokens": usage.output,
        }
    )


@hookimpl
def register_routes():
    return [
        (r"^/-/llm-usage-simple-prompt$", llm_usage_simple_prompt),
    ]


@hookimpl
def startup(datasette):
    async def inner():
        await get_database(datasette, migrate=True)

    return inner


async def get_database(datasette, migrate=False):
    plugin_config = datasette.plugin_config("datasette-llm-usage") or {}
    db_name = plugin_config.get("database")
    if db_name:
        db = datasette.get_database(db_name)
    else:
        db = datasette.get_internal_database()
    if migrate:
        await db.execute_write_fn(lambda conn: migration.apply(Database(conn)))
    return db


async def subtract_credits(db, purpose, model_id, input_tokens, output_tokens):
    price = next(p for p in PRICES if p.model_id == model_id)
    cost = price.cost_in_credits(input_tokens, output_tokens)
    id = None
    print(
        "Subtract credits", type(purpose), model_id, input_tokens, output_tokens, cost
    )
    if purpose:
        # Is there a purpose row?
        sql = "select id from _llm_allowance where purpose = :purpose"
        row = (await db.execute(sql, {"purpose": purpose})).first()
        print("first attempt row=", row)
        if row:
            id = row["id"]
    else:
        # use the purpose is null row instead
        sql = "select id from _llm_allowance where purpose is null"
        id = (await db.execute(sql)).single_value()
    await db.execute(
        "update _llm_allowance set credits_remaining = credits_remaining - :cost where id = :id",
        {
            "cost": cost,
            "id": id,
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
            await db.execute_write(
                """
            insert into _llm_usage (created, model, purpose, actor_id, input_tokens, output_tokens)
            values (:created, :model, {purpose}, {actor_id}, :input_tokens, :output_tokens)
            """.format(
                    actor_id=":actor_id" if actor_id else "null",
                    purpose=":purpose" if self.purpose else "null",
                ),
                {
                    "created": int(time.time() * 1000),
                    "model": self.model.model_id,
                    "purpose": self.purpose,
                    "actor_id": actor_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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

    def get_async_models(self):
        return [WrappedModel(model, self.datasette) for model in llm.get_async_models()]

    def get_async_model(self, model_id=None, purpose=None):
        return WrappedModel(
            llm.get_async_model(model_id), self.datasette, purpose=purpose
        )

    async def has_allowance(self, purpose: Optional[str] = None):
        db = self.datasette.get_database()
        if purpose:
            #  First check allowance for this purpose
            sql = (
                "select credits_remaining from _llm_allowance where purpose = :purpose"
            )
            credits_remaining = (
                await db.execute(sql, {"purpose": purpose})
            ).single_value()
            if credits_remaining > 0:
                return True
        # Check general allowance instead
        credits_remaining = await db.execute(
            "select credits_remaining from _llm_allowance where purpose is null",
        ).single_value()
        return credits_remaining > 0
