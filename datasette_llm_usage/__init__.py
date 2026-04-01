from __future__ import annotations
from contextlib import asynccontextmanager
from datasette import hookimpl, Response
from datasette.permissions import Action
from datasette.utils.asgi import Forbidden
from datasette_llm_usage.migrations import migration
from sqlite_utils import Database
import json
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
                tool_calls = await response.tool_calls()
                tool_calls_json = (
                    json.dumps(
                        [
                            {
                                "name": tc.name,
                                "arguments": tc.arguments,
                                "tool_call_id": tc.tool_call_id,
                            }
                            for tc in tool_calls
                        ]
                    )
                    if tool_calls
                    else None
                )
                tool_results_json = (
                    json.dumps(
                        [
                            {
                                "name": tr.name,
                                "output": tr.output,
                                "tool_call_id": tr.tool_call_id,
                            }
                            for tr in response.prompt.tool_results
                        ]
                    )
                    if response.prompt.tool_results
                    else None
                )
                await db.execute_write(
                    """
                    insert into llm_usage_prompt_log
                        (created, model, purpose, actor_id, prompt, response,
                         input_tokens, output_tokens, tool_calls, tool_results)
                    values
                        (:created, :model, :purpose, :actor_id, :prompt, :response,
                         :input_tokens, :output_tokens, :tool_calls, :tool_results)
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
                        "tool_calls": tool_calls_json,
                        "tool_results": tool_results_json,
                    },
                )

        await result.on_response_done(on_complete)

    return usage_tracker


async def llm_usage_simple_prompt(datasette, request):
    if not await datasette.allowed(
        actor=request.actor, action="llm-usage-simple-prompt"
    ):
        raise Forbidden("Permission denied")
    from datasette_llm import LLM

    llm_instance = LLM(datasette)
    all_models = await llm_instance.models(purpose="simple_prompt")
    model_ids = [m.model_id for m in all_models]

    prompt_text = request.args.get("prompt")
    selected_model = request.args.get("model") or (model_ids[0] if model_ids else None)
    context = {
        "models": model_ids,
        "selected_model": selected_model,
        "prompt_text": prompt_text,
    }

    if prompt_text and selected_model:
        model = await llm_instance.model(
            selected_model, purpose="simple_prompt", actor=request.actor
        )
        response = await model.prompt(prompt_text)
        text = await response.text()
        usage = await response.usage()
        context["response_text"] = text
        context["input_tokens"] = usage.input
        context["output_tokens"] = usage.output

    return Response.html(
        await datasette.render_template(
            "llm_usage_simple_prompt.html", context=context, request=request
        )
    )


@hookimpl
def register_actions(datasette):
    return [
        Action(
            name="llm-usage-simple-prompt",
            description="Access the simple prompt endpoint",
        ),
    ]


@hookimpl
def register_routes():
    return [
        (r"^/-/llm-usage-simple-prompt$", llm_usage_simple_prompt),
    ]
