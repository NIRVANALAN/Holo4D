"""Select a canonical Morton resolution from calibration summaries."""

from __future__ import annotations

def recommend_bits(
    summary: list[dict],
    *,
    min_cross_query_merge_ratio: float,
    max_total_collapse_ratio: float,
) -> dict:
    """Choose the finest tested tolerance that binds without over-collapsing.

    A bit value is eligible only when every requested dataset reaches the
    minimum cross-query merge rate and stays below the total-collapse ceiling.
    Choosing the largest eligible bit count keeps the tolerance as tight as
    possible. These thresholds are a preflight contract, not a learned optimum.
    """
    datasets = sorted({row["dataset"] for row in summary})
    bit_values = sorted({int(row["bits"]) for row in summary}, reverse=True)
    eligible = []
    diagnostics = []
    for bits in bit_values:
        rows = [row for row in summary if int(row["bits"]) == bits]
        by_dataset = {row["dataset"]: row for row in rows}
        passes = len(by_dataset) == len(datasets) and all(
            row["mean_cross_query_merge_ratio"] >= min_cross_query_merge_ratio
            and row["mean_total_collapse_ratio"] <= max_total_collapse_ratio
            for row in rows
        )
        diagnostics.append(
            {
                "bits": bits,
                "passes": passes,
                "datasets": {
                    name: {
                        "mean_cross_query_merge_ratio": by_dataset[name][
                            "mean_cross_query_merge_ratio"
                        ],
                        "mean_total_collapse_ratio": by_dataset[name][
                            "mean_total_collapse_ratio"
                        ],
                    }
                    for name in datasets
                    if name in by_dataset
                },
            }
        )
        if passes:
            eligible.append(bits)
    return {
        "recommended_bits": max(eligible) if eligible else None,
        "rule": "largest bits passing every dataset",
        "min_cross_query_merge_ratio": min_cross_query_merge_ratio,
        "max_total_collapse_ratio": max_total_collapse_ratio,
        "diagnostics": diagnostics,
    }
