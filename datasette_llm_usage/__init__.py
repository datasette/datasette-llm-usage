from __future__ import annotations
from contextlib import asynccontextmanager
from datasette import hookimpl, Response
from datasette_llm_usage.migrations import migration
from sqlite_utils import Database
import time


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


@hookimpl
def startup(datasette):
    async def inner():
        await get_database(datasette, migrate=True)

    return inner


@hookimpl
def llm_prompt_context(datasette, model_id, prompt, purpose, actor):
    @asynccontextmanager
    async def usage_tracker(result):
        yield

        async def on_complete(response):
            usage = await response.usage()
            input_tokens = usage.input if usage else 0
            output_tokens = usage.output if usage else 0
            db = await get_database(datasette)
            actor_id = actor.get("id") if actor else None

            await db.execute_write(
                """
                insert into llm_usage (created, model, purpose, actor_id, input_tokens, output_tokens)
                values (:created, :model, :purpose, :actor_id, :input_tokens, :output_tokens)
                """,
                {
                    "created": int(time.time() * 1000),
                    "model": model_id,
                    "purpose": purpose,
                    "actor_id": actor_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

            plugin_config = datasette.plugin_config("datasette-llm-usage") or {}
            if plugin_config.get("log_prompts"):
                response_text = await response.text()
                await db.execute_write(
                    """
                    insert into llm_usage_prompt_log (created, model, purpose, actor_id, prompt, response, input_tokens, output_tokens)
                    values (:created, :model, :purpose, :actor_id, :prompt, :response, :input_tokens, :output_tokens)
                    """,
                    {
                        "created": int(time.time() * 1000),
                        "model": model_id,
                        "purpose": purpose,
                        "actor_id": actor_id,
                        "prompt": prompt,
                        "response": response_text,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    },
                )

        await result.response.on_done(on_complete)

    return usage_tracker


async def llm_usage_simple_prompt(datasette, request):
    if not request.actor:
        return Response.text("Not logged in", status=403)
    from datasette_llm import LLM

    llm = LLM(datasette)
    prompt_text = request.args.get("prompt")
    if not prompt_text:
        return Response.html("<form><input name=prompt><button>Submit</button></form>")
    model = await llm.model("gpt-4o-mini", purpose="simple_prompt", actor=request.actor)
    response = await model.prompt(prompt_text)
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
