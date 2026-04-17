from __future__ import annotations

from pathlib import Path

from part1_network_utils import (
    build_editor_network,
    ensure_dir,
    load_wikidata_talk_data,
    resolve_dataset_path,
    sample_grouped_discussions,
    sample_node_mapping,
    save_group_size_distribution,
    save_large_network_overview,
    save_medium_network_overview,
    save_small_network_overview,
    summarise_raw_and_graph,
    top_degree_nodes,
    top_weighted_edges,
)


DATASET_FILENAME = "BOT_REQUESTS.csv"
SCALE_LABEL = "small"
NETWORK_VIEW_MODE = "small"  # one of: small, medium, large


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_path = resolve_dataset_path(DATASET_FILENAME, script_dir=script_dir)
    output_dir = ensure_dir(script_dir / "outputs" / f"part1_{SCALE_LABEL}_task_a")

    raw_df, clean_df = load_wikidata_talk_data(dataset_path)
    G, username_to_id, id_to_username, grouped_discussions = build_editor_network(clean_df)

    summary_df = summarise_raw_and_graph(raw_df, clean_df, grouped_discussions, G)
    top_degree_df = top_degree_nodes(G, top_n=20)
    top_edges_df = top_weighted_edges(G, top_n=20)
    grouped_sample_df = sample_grouped_discussions(grouped_discussions, top_n=20)
    node_mapping_df = sample_node_mapping(username_to_id, top_n=30)

    prefix = f"part1_{SCALE_LABEL}_task_a"
    summary_df.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    top_degree_df.to_csv(output_dir / f"{prefix}_top_degree_nodes.csv", index=False)
    top_edges_df.to_csv(output_dir / f"{prefix}_top_weighted_edges.csv", index=False)
    grouped_sample_df.to_csv(output_dir / f"{prefix}_sample_grouped_discussions.csv", index=False)
    node_mapping_df.to_csv(output_dir / f"{prefix}_sample_node_mapping.csv", index=False)

    save_group_size_distribution(
        grouped_discussions,
        output_dir / f"{prefix}_group_size_distribution.png",
        title=f"{dataset_path.stem}: participants per page-thread discussion",
    )

    if NETWORK_VIEW_MODE == "small":
        save_small_network_overview(
            G,
            output_dir / f"{prefix}_network_overview.png",
            title=f"{dataset_path.stem}: editor network overview",
        )
    elif NETWORK_VIEW_MODE == "medium":
        save_medium_network_overview(
            G,
            output_dir / f"{prefix}_network_overview.png",
            title=f"{dataset_path.stem}: largest connected component overview",
        )
    else:
        save_large_network_overview(
            G,
            output_dir / f"{prefix}_network_overview.png",
            title=f"{dataset_path.stem}: top-degree subgraph overview",
        )

    print("=" * 72)
    print(f"Part 1 {SCALE_LABEL.title()} Task A completed for: {dataset_path.name}")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("\nTop degree nodes:")
    print(top_degree_df.head(10).to_string(index=False))
    print("\nTop weighted edges:")
    print(top_edges_df.head(10).to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
