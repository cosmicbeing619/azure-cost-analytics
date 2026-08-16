"""Builds/updates a stable alias table for hashed ID columns
(SubscriptionName, ResourceGroup, ResourceName) so labels stay
consistent as new data arrives — existing IDs keep their alias,
only genuinely new IDs get a new one appended.
"""
import json
from pathlib import Path
import pandas as pd

ALIAS_PATH = Path(__file__).resolve().parent.parent / "docs" / "data" / "alias_map.json"

COLUMN_PREFIXES = {
    "SubscriptionName": "Sub",
    "ResourceGroup": "RG",
    "ResourceName": "Resource",
}


def _load_existing() -> dict:
    if ALIAS_PATH.exists():
        return json.loads(ALIAS_PATH.read_text())
    return {}


def build_or_update_aliases(df: pd.DataFrame) -> dict:
    """Returns the full alias map: {column: {raw_id: alias_label}}.
    Ranks by total cost so Sub-01 is always the highest-spend entity —
    but only assigns NEW aliases to IDs not already in the saved map,
    so existing labels never shift when new data is added.
    """
    all_aliases = _load_existing()

    for column, prefix in COLUMN_PREFIXES.items():
        existing = all_aliases.get(column, {})
        ranked = (
            df.groupby(column)["CostInBillingCurrency"]
            .sum()
            .sort_values(ascending=False)
            .index
        )
        next_idx = len(existing) + 1
        for entity_id in ranked:
            if entity_id not in existing:
                existing[entity_id] = f"{prefix}-{next_idx:02d}"
                next_idx += 1
        all_aliases[column] = existing

    ALIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ALIAS_PATH.write_text(json.dumps(all_aliases, indent=2))
    return all_aliases


def apply_aliases(df: pd.DataFrame, alias_map: dict) -> pd.DataFrame:
    """Adds *_alias columns to the dataframe for display purposes,
    leaving the original hashed columns untouched (still your real join keys)."""
    df = df.copy()
    for column in COLUMN_PREFIXES:
        df[f"{column}_alias"] = df[column].map(alias_map.get(column, {}))
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import load_all
    from clean import clean

    df = load_all()
    df = clean(df)
    alias_map = build_or_update_aliases(df)
    df = apply_aliases(df, alias_map)

    for column in COLUMN_PREFIXES:
        n = len(alias_map[column])
        print(f"{column}: {n} unique entities aliased")
    print("\nSample:")
    print(df[["SubscriptionName_alias", "ResourceGroup_alias", "ResourceName_alias", "CostInBillingCurrency"]].head())
    print(f"\nAlias map saved to: {ALIAS_PATH}")