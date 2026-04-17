from __future__ import annotations

from pathlib import Path

from part1_network_utils import (
    analyse_propagation_scenario,
    build_editor_network,
    choose_suspicious_sources,
    ensure_dir,
    load_wikidata_talk_data,
    resolve_dataset_path,
    save_priority_focus_network,
    save_priority_scores_plot,
)


DATASET_FILENAME = "BOT_REQUESTS.csv"
SCALE_LABEL = "small"
TOP_K_PRIORITY = 25
TOP_K_PLOT = 15


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    dataset_path = resolve_dataset_path(DATASET_FILENAME, script_dir=script_dir)
    output_dir = ensure_dir(script_dir / "outputs" / f"part1_{SCALE_LABEL}_task_c")

    raw_df, clean_df = load_wikidata_talk_data(dataset_path)
    G, username_to_id, id_to_username, grouped_discussions = build_editor_network(clean_df)

    single_source, pair_sources, selection_df = choose_suspicious_sources(G)

    single_summary_df, single_one_hop_df, single_top_df, single_full_df = analyse_propagation_scenario(
        G=G,
        sources=[single_source],
        username_to_id=username_to_id,
        top_k=TOP_K_PRIORITY,
        scenario_name="single_source",
    )
    dual_summary_df, dual_one_hop_df, dual_top_df, dual_full_df = analyse_propagation_scenario(
        G=G,
        sources=pair_sources,
        username_to_id=username_to_id,
        top_k=TOP_K_PRIORITY,
        scenario_name="dual_source",
    )

    prefix = f"part1_{SCALE_LABEL}_task_c"
    selection_df.to_csv(output_dir / f"{prefix}_source_selection.csv", index=False)
    single_summary_df.to_csv(output_dir / f"{prefix}_single_source_summary.csv", index=False)
    single_one_hop_df.to_csv(output_dir / f"{prefix}_single_source_one_hop_exposure.csv", index=False)
    single_top_df.to_csv(output_dir / f"{prefix}_single_source_priority_top25.csv", index=False)
    single_full_df.to_csv(output_dir / f"{prefix}_single_source_priority_full.csv", index=False)
    dual_summary_df.to_csv(output_dir / f"{prefix}_dual_source_summary.csv", index=False)
    dual_one_hop_df.to_csv(output_dir / f"{prefix}_dual_source_one_hop_exposure.csv", index=False)
    dual_top_df.to_csv(output_dir / f"{prefix}_dual_source_priority_top25.csv", index=False)
    dual_full_df.to_csv(output_dir / f"{prefix}_dual_source_priority_full.csv", index=False)

    save_priority_scores_plot(
        single_top_df,
        [single_source],
        output_dir / f"{prefix}_single_source_priority_scores.png",
        title=f"{dataset_path.stem}: single-source priority list",
        top_n=TOP_K_PLOT,
    )
    save_priority_focus_network(
        G,
        username_to_id,
        [single_source],
        single_top_df,
        output_dir / f"{prefix}_single_source_priority_network.png",
        title=f"{dataset_path.stem}: single-source priority network",
        top_n=TOP_K_PLOT,
    )
    save_priority_scores_plot(
        dual_top_df,
        pair_sources,
        output_dir / f"{prefix}_dual_source_priority_scores.png",
        title=f"{dataset_path.stem}: dual-source priority list",
        top_n=TOP_K_PLOT,
    )
    save_priority_focus_network(
        G,
        username_to_id,
        pair_sources,
        dual_top_df,
        output_dir / f"{prefix}_dual_source_priority_network.png",
        title=f"{dataset_path.stem}: dual-source priority network",
        top_n=TOP_K_PLOT,
    )

    print("=" * 72)
    print(f"Part 1 {SCALE_LABEL.title()} Task C completed for: {dataset_path.name}")
    print("=" * 72)
    print("\nSource selection:")
    print(selection_df.to_string(index=False))
    print("\nSingle-source summary:")
    print(single_summary_df.to_string(index=False))
    print("\nTop 10 single-source priority editors:")
    print(single_top_df.head(10).to_string(index=False))
    print("\nDual-source summary:")
    print(dual_summary_df.to_string(index=False))
    print("\nTop 10 dual-source priority editors:")
    print(dual_top_df.head(10).to_string(index=False))
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
