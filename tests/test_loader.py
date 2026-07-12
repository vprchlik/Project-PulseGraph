"""Tests for data loading utilities."""


from pulsegraph.data.loader import (
    build_dataset_for_forecasting,
    generate_synthetic_data,
    prepare_star_series,
)


def test_synthetic_data_shape():
    df = generate_synthetic_data(n_repos=10, n_days=100, seed=42)
    assert "repo_name" in df.columns
    assert "event_date" in df.columns
    assert "stars" in df.columns
    assert len(df["repo_name"].unique()) == 10


def test_prepare_star_series():
    df = generate_synthetic_data(n_repos=5, n_days=200, seed=42)
    repo = df["repo_name"].unique()[0]
    series = prepare_star_series(df, repo, min_history_days=30)
    assert series is not None
    assert len(series) >= 30


def test_prepare_star_series_insufficient_history():
    df = generate_synthetic_data(n_repos=1, n_days=10, seed=42)
    repo = df["repo_name"].unique()[0]
    series = prepare_star_series(df, repo, min_history_days=30)
    assert series is None


def test_build_dataset():
    df = generate_synthetic_data(n_repos=10, n_days=200, seed=42)
    repos = df["repo_name"].unique().tolist()
    dataset = build_dataset_for_forecasting(df, repos, min_history_days=30)
    assert len(dataset) > 0
    for name, series in dataset.items():
        assert len(series) >= 30
