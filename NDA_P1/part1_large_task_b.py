from __future__ import annotations

from pathlib import Path

from part1_network_utils import (
    build_editor_network,
    build_observed_vs_random_table,
    centrality_table_for_reporting,
    compute_observed_metrics,
    compute_random_graph_baseline,
    ensure_dir,
    load_wikidata_talk_data,
    resolve_dataset_path,
    save_degree_histogram,
    save_degree_rank_plot,
    save_observed_vs_random_plot,
    summarise_random_baseline,
)


DATASET_FILENAME = "REQUEST_FOR_DELETION.csv"
SCALE_LABEL = "large"
RANDOM_BASELINE_RUNS = 20


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_path = resolve_dataset_path(DATASET_FILENAME, script_dir=script_dir)
    output_dir = ensure_dir(script_dir / "outputs" / f"part1_{SCALE_LABEL}_task_b")

    raw_df, clean_df = load_wikidata_talk_data(dataset_path)
    G, username_to_id, id_to_username, grouped_discussions = build_editor_network(clean_df)

    observed_df, lcc_graph, path_method = compute_observed_metrics(G)
    random_df = compute_random_graph_baseline(
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_graphs=RANDOM_BASELINE_RUNS,
    )
    random_summary_df = summarise_random_baseline(random_df)
    comparison_df = build_observed_vs_random_table(observed_df, random_summary_df)
    centrality_df = centrality_table_for_reporting(G, top_n=20)

    prefix = f"part1_{SCALE_LABEL}_task_b"
    observed_df.to_csv(output_dir / f"{prefix}_observed_metrics.csv", index=False)
    random_df.to_csv(output_dir / f"{prefix}_random_graph_metrics.csv", index=False)
    random_summary_df.to_csv(output_dir / f"{prefix}_random_baseline_summary.csv", index=False)
    comparison_df.to_csv(output_dir / f"{prefix}_observed_vs_random.csv", index=False)
    centrality_df.to_csv(output_dir / f"{prefix}_top_structural_nodes.csv", index=False)

    save_degree_histogram(
        G,
        output_dir / f"{prefix}_degree_histogram.png",
        title=f"{dataset_path.stem}: degree distribution",
    )
    save_degree_rank_plot(
        G,
        output_dir / f"{prefix}_degree_rank_plot.png",
        title=f"{dataset_path.stem}: degree-rank plot",
    )
    save_observed_vs_random_plot(
        observed_df,
        random_summary_df,
        output_dir / f"{prefix}_observed_vs_random.png",
        title=f"{dataset_path.stem}: observed network vs random baseline",
    )

    print("=" * 72)
    print(f"Part 1 {SCALE_LABEL.title()} Task B completed for: {dataset_path.name}")
    print("=" * 72)
    print("\nObserved metrics:")
    print(observed_df.to_string(index=False))
    print("\nRandom baseline summary:")
    print(random_summary_df.to_string(index=False))
    print("\nTop structural nodes:")
    print(centrality_df.head(10).to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
