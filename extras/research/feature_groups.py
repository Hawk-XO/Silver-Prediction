"""
research/feature_groups.py

Groups the ~20-30 engineered feature columns (features/pipeline.get_feature_columns
output) into a handful of logical clusters, so the feature-search harness can
toggle related columns on/off as a unit (e.g. "drop all EMA features") rather
than managing every individual column. Full combinatorial search across
7-8 groups is 2^7-2^8 combos x hundreds of walk-forward folds each -- far
too expensive -- so this module is what makes leave-one-group-out /
single-group-only search (~2n combos) tractable.

Grouping is prefix/pattern-based against whatever columns are ACTUALLY
present in a given feature matrix (not a hardcoded list), so it stays
correct if features/indicators.py etc. change their windows or a new
indicator gets added later -- the new column just needs a matching pattern
added below, or it silently lands in "other" (visible, not lost).

Groups (matches features/indicators.py, rolling_stats.py, cross_asset.py,
calendar_features.py column-naming conventions):
    ema             ema_9, ema_21, ema_50, ema_200, ...
    macd            macd, macd_signal, macd_diff
    rsi             rsi_14, ...
    atr             atr_14, ...
    bollinger       bb_percent_b_20, ...
    adx             adx_14, ...
    return_stats    ret_mean_5, ret_std_5, ret_skew_5, ...
    cross_asset     comex_mcx_spread, comex_mcx_spread_z_10, gold_silver_ratio
    calendar        day_of_week, days_to_expiry
    other           anything not matched above (kept visible, never dropped silently)
"""

from __future__ import annotations

import re

# Ordered (first match wins) prefix/regex rules. Order matters because e.g.
# "comex_mcx_spread_z_10" must NOT be caught by a generic "comex" rule if we
# ever add one for something else.
_GROUP_RULES: list[tuple[str, re.Pattern]] = [
    ("ema", re.compile(r"^ema_\d+$")),
    ("macd", re.compile(r"^macd(_signal|_diff)?$")),
    ("rsi", re.compile(r"^rsi_\d+$")),
    ("atr", re.compile(r"^atr_\d+$")),
    ("bollinger", re.compile(r"^bb_percent_b_\d+$")),
    ("adx", re.compile(r"^adx_\d+$")),
    ("return_stats", re.compile(r"^ret_(mean|std|skew)_\d+$")),
    ("cross_asset", re.compile(r"^(comex_mcx_spread(_z_\d+)?|gold_silver_ratio)$")),
    ("calendar", re.compile(r"^(day_of_week|days_to_expiry)$")),
]


def group_feature_columns(feature_cols: list[str]) -> dict[str, list[str]]:
    """
    Map each column in `feature_cols` to its logical group name. Returns
    {group_name: [columns]}, preserving the input order within each group.
    Columns matching no rule land in "other" -- check this key rather than
    assuming every column got grouped, especially after feature-pipeline
    changes.
    """
    groups: dict[str, list[str]] = {name: [] for name, _ in _GROUP_RULES}
    groups["other"] = []

    for col in feature_cols:
        matched = False
        for name, pattern in _GROUP_RULES:
            if pattern.match(col):
                groups[name].append(col)
                matched = True
                break
        if not matched:
            groups["other"].append(col)

    # Drop empty groups (e.g. "other" when everything matched, or a group
    # whose feature type isn't present in this particular feature matrix).
    return {name: cols for name, cols in groups.items() if cols}


def leave_one_group_out_subsets(feature_cols: list[str]) -> dict[str, list[str]]:
    """
    Return {"drop_<group>": [remaining columns]} for each group -- one
    subset per group, with that group's columns removed and everything
    else kept. Answers "does removing this group hurt or help?" without
    full combinatorial search.
    """
    groups = group_feature_columns(feature_cols)
    subsets = {}
    for name in groups:
        subsets[f"drop_{name}"] = [c for c in feature_cols if c not in groups[name]]
    return subsets


def single_group_only_subsets(feature_cols: list[str]) -> dict[str, list[str]]:
    """
    Return {"only_<group>": [that group's columns]} for each group -- the
    complementary view to leave_one_group_out: "how much signal does THIS
    group alone carry?"
    """
    groups = group_feature_columns(feature_cols)
    return {f"only_{name}": cols for name, cols in groups.items()}


def all_features_subset(feature_cols: list[str]) -> dict[str, list[str]]:
    """The baseline candidate: every engineered feature, ungrouped. Always
    include this as a comparison point against the group-ablation subsets."""
    return {"all_features": list(feature_cols)}


def default_search_candidates(feature_cols: list[str]) -> dict[str, list[str]]:
    """
    The standard ~2n+1 candidate set the CLI uses by default: baseline
    (all features) + leave-one-group-out + single-group-only. Callers
    wanting full combinatorial search should build their own dict and pass
    it to research.search_harness.run_search() instead (see that module's
    warning about multiple-comparisons blowup before doing so).
    """
    candidates = {}
    candidates.update(all_features_subset(feature_cols))
    candidates.update(leave_one_group_out_subsets(feature_cols))
    candidates.update(single_group_only_subsets(feature_cols))
    return candidates
