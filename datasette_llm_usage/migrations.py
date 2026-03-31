from sqlite_migrate import Migrations

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
    # Legacy migration, kept for compatibility
    db["_llm_allowance"].create(
        {
            "id": int,
            "created": int,
            "credits_remaining": int,
            "daily_reset": bool,
            "daily_reset_amount": int,
            "purpose": str,
        },
        pk="id",
        not_null=("id", "created", "credits_remaining"),
    )


@migration()
def create_prompt_log_table(db):
    db["_llm_prompt_log"].create(
        {
            "id": int,
            "created": int,
            "model": str,
            "purpose": str,
            "actor_id": str,
            "prompt": str,
            "response": str,
            "input_tokens": int,
            "output_tokens": int,
        },
        pk="id",
    )


@migration()
def drop_allowance_table(db):
    db["_llm_allowance"].drop(ignore=True)


@migration()
def rename_tables(db):
    if "_llm_usage" in db.table_names():
        db.rename_table("_llm_usage", "llm_usage")
    if "_llm_prompt_log" in db.table_names():
        db.rename_table("_llm_prompt_log", "llm_usage_prompt_log")
