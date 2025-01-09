from __future__ import annotations
from dataclasses import dataclass
from datasette import hookimpl, Response
import json
import time
from typing import Optional, Iterable
import llm
from .pricing import calculate_costs
from .db import migration, get_database


class TokensExhausted(Exception):
    pass


@dataclass
class ModelAllowance:
    model_id: str
    provider: str
    credits_remaining: int
    tokens_remaining: int


async def subtract_credits(db, model_id, input_tokens, output_tokens):
    row = (
        await db.execute(
            """
        SELECT id, provider_id FROM _llm_usage_model WHERE name = ?
        """,
            [model_id],
        )
    ).first()
    if not row:
        raise ValueError("Unknown model_id: {}".format(model_id))
    provider_db_id = row["provider_id"]

    # Check/do daily reset before checking available credits
    await _maybe_daily_reset(db, provider_db_id)

    # Calculate cost
    input_cost, output_cost = await calculate_costs(
        db, model_id, input_tokens, output_tokens
    )
    cost = int((input_cost + output_cost) * 1_000_000)  # Convert to credits

    # Check if we have enough credits
    credits_remaining = (
        await db.execute(
            """
        SELECT credits_remaining FROM _llm_usage_allowance
        WHERE provider_id = ?
        """,
            [provider_db_id],
        )
    ).first()["credits_remaining"]

    if credits_remaining < cost:
        raise TokensExhausted("Not enough credits remaining")

    # Subtract credits
    await db.execute_write(
        """
        UPDATE _llm_usage_allowance
        SET credits_remaining = credits_remaining - :cost
        WHERE provider_id = :provider_id
        """,
        {
            "provider_id": provider_db_id,
            "cost": cost,
        },
    )


class WrappedModel:
    def __init__(
        self,
        model: llm.AsyncModel,
        datasette,
        provider: str,
        purpose: Optional[str] = None,
    ):
        self.model = model
        self.datasette = datasette
        self.provider = provider
        self.purpose = purpose

    async def prompt(
        self,
        prompt: Optional[str],
        system: Optional[str] = None,
        actor_id: Optional[str] = None,
        **kwargs,
    ):
        # Check if there are enough tokens in the allowance
        llm = LLM(self.datasette)
        if not await llm.has_allowance(self):
            raise TokensExhausted("No tokens remaining")

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
            await subtract_credits(db, self.model.model_id, input_tokens, output_tokens)

        await response.on_done(done)
        return response

    def __repr__(self):
        return f"WrappedModel: {self.model.model_id}"


class LLM:
    def __init__(self, datasette):
        self.datasette = datasette

    async def get_async_models(self, model_ids: Optional[Iterable[str]] = None):
        models = llm.get_async_models()
        models_by_id = {model.model_id: model for model in models}
        # Filter for the ones in the database and add provider
        db = await get_database(self.datasette)
        sql = """
            select _llm_usage_model.name as model_id, _llm_usage_provider.name as provider
            from _llm_usage_model
            join _llm_usage_provider on _llm_usage_model.provider_id = _llm_usage_provider.id
        """
        params = []
        if model_ids:
            sql += " where _llm_usage_model.name in ({})".format(
                ",".join("?" for _ in model_ids)
            )
            params = list(model_ids)
        rows = await db.execute(sql, params)
        wrapped_models = []
        for row in rows.rows:
            model_id = row["model_id"]
            provider = row["provider"]
            model = models_by_id.get(model_id)
            if model:
                wrapped_models.append(WrappedModel(model, self.datasette, provider))
        return wrapped_models

    async def get_async_model(self, model_id=None, purpose=None):
        models = await self.get_async_models(model_ids=[model_id])
        model = models[0]
        model.purpose = purpose
        return model

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

    async def has_allowance(self, wrapped_model: WrappedModel):
        db = await get_database(self.datasette)
        # Check general allowance for this  model
        credits_remaining = (
            await db.execute(
                """
            select credits_remaining from _llm_usage_allowance
            where provider_id = (select id from _llm_usage_provider where name = :provider)
            """,
                {"provider": wrapped_model.provider},
            )
        ).single_value()
        return credits_remaining > 0


async def llm_usage_simple_prompt(datasette, request):
    if not request.actor:
        return Response.text("Not logged in", status=403)
    llm = LLM(datasette)
    post_data = await request.post_vars()
    prompt = post_data.get("prompt")
    context = {}
    if prompt:
        model_id = post_data.get("model") or "gpt-4o-mini"
        model = await llm.get_async_model(model_id, purpose="simple_prompt")
        try:
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
        except TokensExhausted:
            context["error"] = "No tokens remaining"

    context["allowances"] = await llm.get_model_allowances()

    return Response.html(
        await datasette.render_template(
            "llm_usage_single_prompt.html", context, request=request
        )
    )


async def llm_usage_credits(datasette, request):
    llm = LLM(datasette)
    allowances = await llm.get_model_allowances()
    providers_dict = {}
    for allowance in allowances:
        providers_dict.setdefault(allowance.provider, []).append(allowance)
    providers = [
        {"provider": key, "models": value} for key, value in providers_dict.items()
    ]
    providers.sort(key=lambda x: x["provider"])
    return Response.html(
        await datasette.render_template(
            "llm_usage_credits.html", {"providers": providers}, request=request
        )
    )


@hookimpl
def register_routes():
    return [
        (r"^/-/llm-usage-simple-prompt$", llm_usage_simple_prompt),
        (r"^/-/llm-usage-credits$", llm_usage_credits),
    ]


@hookimpl
def startup(datasette):
    async def inner():
        await get_database(datasette, migrate=True, populate=True)

    return inner


async def _maybe_daily_reset(db, provider_id: int):
    """
    If last_reset_timestamp is at least 24 hours old, reset credits_remaining
    to daily_reset_amount and set last_reset_timestamp = int(time.time()).
    """
    row = (
        await db.execute(
            """
        SELECT credits_remaining, last_reset_timestamp, daily_reset_amount
        FROM _llm_usage_allowance
        WHERE provider_id = :provider_id
        """,
            {"provider_id": provider_id},
        )
    ).first()
    if not row:
        return  # No row for this provider

    daily_reset_amount = row["daily_reset_amount"]
    last_reset_timestamp = row["last_reset_timestamp"] or 0

    if not daily_reset_amount:
        return  # daily_reset_amount is None or 0, so no daily reset feature

    now = int(time.time())
    # Check if at least 24 hours have passed since last reset
    if now - last_reset_timestamp >= 86400:
        await db.execute_write(
            """
            UPDATE _llm_usage_allowance
            SET credits_remaining = :daily_reset_amount,
                last_reset_timestamp = :now
            WHERE provider_id = :provider_id
            """,
            {
                "daily_reset_amount": daily_reset_amount,
                "now": now,
                "provider_id": provider_id,
            },
        )
