"""Build the ``features.signal_map`` table from configs/features.yaml.

The signal_map is a skinny lookup that tells the SQL feature
extractor which (source, itemid) rows map to which feature name,
along with a valid range and optional unit conversion.
"""

from __future__ import annotations

import duckdb
import pandas as pd
from loguru import logger

from aki.utils.config import Config


def build_signal_map(conn: duckdb.DuckDBPyConnection, cfg: Config) -> pd.DataFrame:
    """Materialize ``features.signal_map`` from features.yaml."""
    rows: list[dict] = []

    # Vitals from chartevents
    for v in cfg.features["vitals"]:
        lo, hi = v["valid_range"]
        # Special case: temperature_f gets converted to celsius so its
        # feature_name is also "temperature_c" (not temperature_f) and
        # the valid range is re-expressed on the celsius scale.
        if v["name"] == "temperature_f":
            name, lo, hi, mult, off = "temperature_c", 30.0, 45.0, 5.0 / 9.0, -32.0
        else:
            name, mult, off = v["name"], 1.0, 0.0
        for iid in v["itemids"]:
            rows.append({
                "feature_name": name,
                "source": "chartevents",
                "itemid": iid,
                "valid_low": lo,
                "valid_high": hi,
                "unit_multiplier": mult,
                "unit_offset": off,
            })

    # Labs from labevents
    for lab in cfg.features["labs"]:
        lo, hi = lab["valid_range"]
        for iid in lab["itemids"]:
            rows.append({
                "feature_name": lab["name"],
                "source": "labevents",
                "itemid": iid,
                "valid_low": lo,
                "valid_high": hi,
                "unit_multiplier": 1.0,
                "unit_offset": 0.0,
            })

    df = pd.DataFrame(rows)
    conn.execute("CREATE SCHEMA IF NOT EXISTS features")
    conn.execute("DROP TABLE IF EXISTS features.signal_map")
    conn.register("sig_df", df)
    conn.execute("CREATE TABLE features.signal_map AS SELECT * FROM sig_df")
    conn.unregister("sig_df")
    logger.info(f"features.signal_map: {len(df)} rows ({df['feature_name'].nunique()} features)")
    return df
