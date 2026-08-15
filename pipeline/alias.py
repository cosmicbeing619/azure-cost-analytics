"""Builds/updates a stable alias table for hashed ID columns
(SubscriptionName, ResourceGroup, ResourceName) so labels stay
consistent as new data arrives.
"""
import pandas as pd
import json
from pathlib import Path

ALIAS_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "alias_map.json"

def build_or_update_alias(df: pd.DataFrame, column: str, prefix: str) -> dict:
    existing = {}
    if ALIAS_PATH.exists():
        existing = json.loads(ALIAS_PATH.read_text()).get(column, {})
    ranked = (
        df.groupby(column)["CostInBillingCurrency"].sum().sort_values(ascending=False)
    )
    aliases = dict(existing)
    next_idx = len(aliases) + 1
    for entity_id in ranked.index:
        if entity_id not in aliases:
            aliases[entity_id] = f"{prefix}-{next_idx:02d}"
            next_idx += 1
    return aliases
