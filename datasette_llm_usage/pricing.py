from typing import Dict, List, Optional, Tuple
import json


async def get_model_tiers(db, model_id: str) -> List[Dict]:
    results = await db.execute(
        """
        select tiers from _llm_usage_model
        where name = :model_id
        """,
        {"model_id": model_id},
    )
    row = results.first()
    if not row:
        return []
    return json.loads(row["tiers"])


async def calculate_costs(
    db, model_id: str, input_tokens: int, output_tokens: int
) -> Tuple[float, float]:
    tiers = await get_model_tiers(db, model_id)
    if not tiers:
        return 0.0, 0.0

    # Find the appropriate tier based on input tokens
    selected_tier = None
    for tier in tiers:
        if tier["max_tokens"] is None or input_tokens <= tier["max_tokens"]:
            selected_tier = tier
            break

    if not selected_tier:
        # Use the last tier if no match found
        selected_tier = tiers[-1]

    input_cost = (input_tokens * selected_tier["input_cost"]) / 1_000_000
    output_cost = (output_tokens * selected_tier["output_cost"]) / 1_000_000

    return input_cost, output_cost
