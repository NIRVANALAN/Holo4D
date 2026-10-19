from holo4d.analysis.morton_calibration import recommend_bits


def _row(dataset, bits, merge, collapse):
    return {
        "dataset": dataset,
        "bits": bits,
        "mean_cross_query_merge_ratio": merge,
        "mean_total_collapse_ratio": collapse,
    }


def test_recommendation_chooses_finest_non_inert_non_overcollapsed_bits():
    summary = []
    for dataset in ("kubric", "syn4d"):
        summary.extend(
            [
                _row(dataset, 8, 0.06, 0.07),
                _row(dataset, 7, 0.16, 0.22),
                _row(dataset, 6, 0.31, 0.42),
            ]
        )
    result = recommend_bits(
        summary,
        min_cross_query_merge_ratio=0.10,
        max_total_collapse_ratio=0.35,
    )
    assert result["recommended_bits"] == 7
    assert [row["bits"] for row in result["diagnostics"] if row["passes"]] == [7]


def test_recommendation_requires_every_dataset_to_pass():
    summary = [
        _row("kubric", 7, 0.15, 0.20),
        _row("syn4d", 7, 0.05, 0.20),
    ]
    result = recommend_bits(
        summary,
        min_cross_query_merge_ratio=0.10,
        max_total_collapse_ratio=0.35,
    )
    assert result["recommended_bits"] is None
