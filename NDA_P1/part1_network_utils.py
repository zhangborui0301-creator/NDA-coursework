from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path


REQUIRED_COLUMNS = {"page_name", "thread_subject", "username"}
DEFAULT_RANDOM_SEED = 42
DEFAULT_RANDOM_BASELINE_RUNS = 20
EXACT_PATH_THRESHOLD = 2500
ASPL_SAMPLE_SIZE = 100
TOP_LABEL_COUNT = 15
TOP_SUBGRAPH_NODES = 200


# ============================================================
# Path helpers
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dataset_path(dataset_filename: str, script_dir: Path | None = None) -> Path:
    candidates: list[Path] = []

    raw = Path(dataset_filename)
    if raw.exists():
        return raw.resolve()

    if script_dir is not None:
        candidates.extend(
            [
                script_dir / dataset_filename,
                script_dir / "datasets" / dataset_filename,
                script_dir.parent / "datasets" / dataset_filename,
            ]
        )

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / dataset_filename,
            cwd / "datasets" / dataset_filename,
            Path("/mnt/data") / dataset_filename,
            Path("/mnt/data") / "datasets" / dataset_filename,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"Could not locate dataset: {dataset_filename}")


# ============================================================
# Data loading and network construction
# ============================================================

def load_wikidata_talk_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        raw_df: original three-column dataframe before cleaning duplicates
        clean_df: cleaned dataframe used for graph construction
    """
    csv_path = Path(csv_path)
    raw_df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    raw_df = raw_df[["page_name", "thread_subject", "username"]].copy()

    clean_df = raw_df.dropna(subset=["page_name", "thread_subject", "username"]).copy()
    for col in ["page_name", "thread_subject", "username"]:
        clean_df[col] = clean_df[col].astype(str).str.strip()

    clean_df = clean_df[
        (clean_df["page_name"] != "")
        & (clean_df["thread_subject"] != "")
        & (clean_df["username"] != "")
    ].copy()

    clean_df = clean_df.drop_duplicates().reset_index(drop=True)
    clean_df["discussion_id"] = clean_df["page_name"] + " || " + clean_df["thread_subject"]
    return raw_df.reset_index(drop=True), clean_df



def build_editor_network(
    df: pd.DataFrame,
    node_col: str = "username",
    group_cols: tuple[str, str] = ("page_name", "thread_subject"),
    weighted: bool = True,
) -> tuple[nx.Graph, dict[str, int], dict[int, str], pd.DataFrame]:
    """
    Generic network builder.

    Default coursework definition:
    - node  = editor username
    - edge  = both editors commented in the same page and same thread
    - weight = number of distinct page-thread groups in which the pair co-appeared
    """
    working_df = df[list(group_cols) + [node_col]].drop_duplicates().copy()

    users = sorted(working_df[node_col].unique())
    username_to_id = {username: idx for idx, username in enumerate(users)}
    id_to_username = {idx: username for username, idx in username_to_id.items()}

    grouped_discussions = (
        working_df.groupby(list(group_cols))[node_col]
        .agg(lambda s: sorted(set(s)))
        .reset_index(name="users")
    )
    grouped_discussions["group_size"] = grouped_discussions["users"].str.len()
    grouped_discussions["discussion_id"] = (
        grouped_discussions[group_cols[0]] + " || " + grouped_discussions[group_cols[1]]
    )

    G = nx.Graph()
    comment_count = df.groupby(node_col).size().to_dict()
    page_count = df.groupby(node_col)[group_cols[0]].nunique().to_dict()
    discussion_count = df.groupby(node_col)["discussion_id"].nunique().to_dict()

    for username, node_id in username_to_id.items():
        G.add_node(
            node_id,
            username=username,
            comment_count=int(comment_count.get(username, 0)),
            page_count=int(page_count.get(username, 0)),
            discussion_count=int(discussion_count.get(username, 0)),
        )

    edge_weights: defaultdict[tuple[int, int], int] = defaultdict(int)
    edge_examples: defaultdict[tuple[int, int], list[str]] = defaultdict(list)

    for row in grouped_discussions.itertuples(index=False):
        users_in_discussion = row.users
        if len(users_in_discussion) < 2:
            continue

        node_ids = sorted(username_to_id[u] for u in users_in_discussion)
        discussion_label = f"{row.page_name} || {row.thread_subject}"

        for u, v in combinations(node_ids, 2):
            edge_key = (u, v)
            edge_weights[edge_key] += 1
            if len(edge_examples[edge_key]) < 3:
                edge_examples[edge_key].append(discussion_label)

    for (u, v), weight in edge_weights.items():
        attrs = {
            "weight": int(weight),
            "shared_discussions": int(weight),
            "example_discussions": edge_examples[(u, v)],
        }
        if weighted:
            G.add_edge(u, v, **attrs)
        else:
            G.add_edge(u, v)

    return G, username_to_id, id_to_username, grouped_discussions


# ============================================================
# Task A helpers
# ============================================================

def summarise_raw_and_graph(raw_df: pd.DataFrame, clean_df: pd.DataFrame, grouped_discussions: pd.DataFrame, G: nx.Graph) -> pd.DataFrame:
    degrees = [d for _, d in G.degree()]
    largest_cc_size = 0
    if G.number_of_nodes() > 0:
        largest_cc_size = len(max(nx.connected_components(G), key=len))

    summary = {
        "rows_raw": int(len(raw_df)),
        "rows_after_cleaning": int(len(clean_df)),
        "rows_removed_during_cleaning": int(len(raw_df) - len(clean_df)),
        "unique_pages": int(clean_df["page_name"].nunique()),
        "unique_threads": int(clean_df["thread_subject"].nunique()),
        "unique_discussions": int(clean_df["discussion_id"].nunique()),
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "connected_components": int(nx.number_connected_components(G)) if G.number_of_nodes() else 0,
        "isolated_nodes": int(nx.number_of_isolates(G)) if G.number_of_nodes() else 0,
        "largest_component_size": int(largest_cc_size),
        "average_degree": float(sum(degrees) / len(degrees)) if degrees else 0.0,
        "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "groups_with_single_editor": int((grouped_discussions["group_size"] == 1).sum()),
        "groups_with_2plus_editors": int((grouped_discussions["group_size"] >= 2).sum()),
        "largest_group_size": int(grouped_discussions["group_size"].max()) if len(grouped_discussions) else 0,
        "average_group_size": float(grouped_discussions["group_size"].mean()) if len(grouped_discussions) else 0.0,
    }
    return pd.DataFrame([summary])



def top_degree_nodes(G: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for node, degree in sorted(G.degree(), key=lambda item: item[1], reverse=True)[:top_n]:
        rows.append(
            {
                "node_id": node,
                "username": G.nodes[node]["username"],
                "degree": degree,
                "comment_count": G.nodes[node].get("comment_count", 0),
                "page_count": G.nodes[node].get("page_count", 0),
                "discussion_count": G.nodes[node].get("discussion_count", 0),
            }
        )
    return pd.DataFrame(rows)



def top_weighted_edges(G: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    rows = []
    for u, v, data in sorted(G.edges(data=True), key=lambda item: item[2].get("weight", 1), reverse=True)[:top_n]:
        rows.append(
            {
                "user_1": G.nodes[u]["username"],
                "user_2": G.nodes[v]["username"],
                "weight": data.get("weight", 1),
                "example_discussions": " ; ".join(data.get("example_discussions", [])),
            }
        )
    return pd.DataFrame(rows)



def sample_grouped_discussions(grouped_discussions: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    sample_df = grouped_discussions.sort_values(["group_size", "discussion_id"], ascending=[False, True]).head(top_n).copy()
    sample_df["users"] = sample_df["users"].apply(lambda users: " ; ".join(users))
    return sample_df[["page_name", "thread_subject", "discussion_id", "group_size", "users"]]



def sample_node_mapping(username_to_id: dict[str, int], top_n: int = 30) -> pd.DataFrame:
    rows = [{"node_id": node_id, "username": username} for username, node_id in username_to_id.items()]
    return pd.DataFrame(rows).sort_values("node_id").head(top_n).reset_index(drop=True)



def _label_dict_for_top_nodes(H: nx.Graph, degree_order: list[int], label_count: int = TOP_LABEL_COUNT) -> dict[int, str]:
    chosen = degree_order[: min(label_count, len(degree_order))]
    return {node: H.nodes[node]["username"] for node in chosen if node in H}



def save_group_size_distribution(grouped_discussions: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 5))
    grouped_discussions["group_size"].value_counts().sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel("Distinct editors in one page-thread discussion")
    plt.ylabel("Number of discussions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_small_network_overview(G: nx.Graph, output_path: Path, title: str) -> None:
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=DEFAULT_RANDOM_SEED, k=0.25)
    degrees = dict(G.degree())
    node_sizes = [20 + 8 * degrees[node] for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, alpha=0.75)
    nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.6)
    degree_order = [node for node, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]
    nx.draw_networkx_labels(G, pos, labels=_label_dict_for_top_nodes(G, degree_order), font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_medium_network_overview(G: nx.Graph, output_path: Path, title: str) -> None:
    if G.number_of_nodes() == 0:
        return
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc_nodes).copy()
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(H, seed=DEFAULT_RANDOM_SEED, k=0.18)
    node_sizes = [20 + 10 * math.sqrt(max(H.degree(n), 1)) for n in H.nodes()]
    edge_widths = [0.2 + 0.15 * math.sqrt(max(d.get("weight", 1), 1)) for _, _, d in H.edges(data=True)]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, alpha=0.75)
    nx.draw_networkx_edges(H, pos, alpha=0.08, width=edge_widths)
    degree_order = [node for node, _ in sorted(H.degree(), key=lambda x: x[1], reverse=True)]
    nx.draw_networkx_labels(H, pos, labels=_label_dict_for_top_nodes(H, degree_order), font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_large_network_overview(G: nx.Graph, output_path: Path, title: str, top_n: int = TOP_SUBGRAPH_NODES) -> None:
    degree_order = [node for node, _ in sorted(G.degree(), key=lambda x: x[1], reverse=True)]
    top_nodes = degree_order[: min(top_n, len(degree_order))]
    H = G.subgraph(top_nodes).copy()
    plt.figure(figsize=(13, 10))
    pos = nx.spring_layout(H, seed=DEFAULT_RANDOM_SEED, k=0.30)
    node_sizes = [25 + 8 * H.degree(n) for n in H.nodes()]
    widths = [0.2 + 0.3 * H[u][v].get("weight", 1) for u, v in H.edges()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, alpha=0.80)
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.25)
    nx.draw_networkx_labels(H, pos, labels=_label_dict_for_top_nodes(H, degree_order), font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Task B helpers
# ============================================================

def largest_connected_component_subgraph(G: nx.Graph) -> nx.Graph:
    if G.number_of_nodes() == 0:
        return nx.Graph()
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc_nodes).copy()



def average_shortest_path_and_diameter_connected(G: nx.Graph) -> tuple[float, int, str]:
    if G.number_of_nodes() <= 1:
        return 0.0, 0, "trivial"

    if G.number_of_nodes() <= EXACT_PATH_THRESHOLD:
        A = nx.to_scipy_sparse_array(G, format="csr")
        dist = shortest_path(A, directed=False, unweighted=True)
        finite = dist[np.isfinite(dist)]
        n = G.number_of_nodes()
        avg_shortest_path = finite.sum() / (n * (n - 1))
        diameter = int(finite.max())
        return float(avg_shortest_path), diameter, "exact"

    rng = random.Random(DEFAULT_RANDOM_SEED)
    nodes = list(G.nodes())
    sample_k = min(ASPL_SAMPLE_SIZE, len(nodes))
    sampled_sources = rng.sample(nodes, sample_k)
    total_distance = 0.0
    total_pairs = 0
    for source in sampled_sources:
        lengths = nx.single_source_shortest_path_length(G, source)
        total_distance += float(sum(lengths.values()))
        total_pairs += len(lengths) - 1
    avg_shortest_path = total_distance / total_pairs if total_pairs else 0.0
    try:
        diameter = int(nx.approximation.diameter(G))
    except Exception:
        diameter = int(max(nx.single_source_shortest_path_length(G, sampled_sources[0]).values()))
    return float(avg_shortest_path), diameter, f"estimated_from_{sample_k}_sources"



def betweenness_centrality_for_reporting(G: nx.Graph) -> dict[int, float]:
    if G.number_of_nodes() <= 2000:
        return nx.betweenness_centrality(G, normalized=True)
    if G.number_of_nodes() <= 5000:
        k = min(100, G.number_of_nodes())
        return nx.betweenness_centrality(G, k=k, normalized=True, seed=DEFAULT_RANDOM_SEED)
    return {node: 0.0 for node in G.nodes()}



def compute_observed_metrics(G: nx.Graph) -> tuple[pd.DataFrame, nx.Graph, str]:
    degrees = [degree for _, degree in G.degree()]
    LCC = largest_connected_component_subgraph(G)
    avg_sp, diameter, path_method = average_shortest_path_and_diameter_connected(LCC)

    metrics = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
        "average_degree": float(sum(degrees) / len(degrees)) if degrees else 0.0,
        "median_degree": float(pd.Series(degrees).median()) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "average_clustering": nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0,
        "transitivity": nx.transitivity(G) if G.number_of_nodes() > 0 else 0.0,
        "assortativity": nx.degree_assortativity_coefficient(G) if G.number_of_edges() > 0 else 0.0,
        "connected_components": nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0,
        "isolated_nodes": nx.number_of_isolates(G) if G.number_of_nodes() > 0 else 0,
        "largest_component_size": LCC.number_of_nodes(),
        "average_shortest_path_length_LCC": avg_sp,
        "diameter_LCC": diameter,
        "path_metric_method": path_method,
    }
    return pd.DataFrame([metrics]), LCC, path_method



def compute_random_graph_baseline(n_nodes: int, n_edges: int, n_graphs: int = DEFAULT_RANDOM_BASELINE_RUNS) -> pd.DataFrame:
    rows = []
    for i in range(n_graphs):
        R = nx.gnm_random_graph(n_nodes, n_edges, seed=DEFAULT_RANDOM_SEED + i)
        RLCC = largest_connected_component_subgraph(R)
        avg_sp, diameter, path_method = average_shortest_path_and_diameter_connected(RLCC)
        rows.append(
            {
                "graph_id": i + 1,
                "nodes": R.number_of_nodes(),
                "edges": R.number_of_edges(),
                "density": nx.density(R) if R.number_of_nodes() > 1 else 0.0,
                "average_degree": float(sum(dict(R.degree()).values()) / R.number_of_nodes()) if R.number_of_nodes() else 0.0,
                "median_degree": float(pd.Series([d for _, d in R.degree()]).median()) if R.number_of_nodes() else 0.0,
                "max_degree": int(max(dict(R.degree()).values())) if R.number_of_nodes() else 0,
                "average_clustering": nx.average_clustering(R) if R.number_of_nodes() > 0 else 0.0,
                "transitivity": nx.transitivity(R) if R.number_of_nodes() > 0 else 0.0,
                "assortativity": nx.degree_assortativity_coefficient(R) if R.number_of_edges() > 0 else 0.0,
                "connected_components": nx.number_connected_components(R) if R.number_of_nodes() > 0 else 0,
                "isolated_nodes": nx.number_of_isolates(R) if R.number_of_nodes() > 0 else 0,
                "largest_component_size": RLCC.number_of_nodes(),
                "average_shortest_path_length_LCC": avg_sp,
                "diameter_LCC": diameter,
                "path_metric_method": path_method,
            }
        )
    return pd.DataFrame(rows)



def summarise_random_baseline(random_df: pd.DataFrame) -> pd.DataFrame:
    summary = {}
    for col in random_df.columns:
        if col in {"graph_id", "path_metric_method"}:
            continue
        summary[f"{col}_mean"] = random_df[col].mean()
        summary[f"{col}_std"] = random_df[col].std()
    summary["random_graph_count"] = int(len(random_df))
    return pd.DataFrame([summary])



def build_observed_vs_random_table(observed_df: pd.DataFrame, random_summary_df: pd.DataFrame) -> pd.DataFrame:
    observed = observed_df.iloc[0].to_dict()
    random_summary = random_summary_df.iloc[0].to_dict()
    metrics = [
        "nodes",
        "edges",
        "density",
        "average_degree",
        "median_degree",
        "max_degree",
        "average_clustering",
        "transitivity",
        "assortativity",
        "connected_components",
        "isolated_nodes",
        "largest_component_size",
        "average_shortest_path_length_LCC",
        "diameter_LCC",
    ]
    rows = []
    for metric in metrics:
        rand_mean = random_summary.get(f"{metric}_mean", np.nan)
        rand_std = random_summary.get(f"{metric}_std", np.nan)
        obs = observed.get(metric, np.nan)
        ratio = obs / rand_mean if pd.notna(rand_mean) and rand_mean != 0 else np.nan
        rows.append(
            {
                "metric": metric,
                "observed": obs,
                "random_mean": rand_mean,
                "random_std": rand_std,
                "observed_to_random_mean_ratio": ratio,
            }
        )
    return pd.DataFrame(rows)



def centrality_table_for_reporting(G: nx.Graph, top_n: int = 20) -> pd.DataFrame:
    degree_dict = dict(G.degree())
    degree_centrality = nx.degree_centrality(G)
    betweenness = betweenness_centrality_for_reporting(G)
    rows = []
    for node in G.nodes():
        rows.append(
            {
                "node_id": node,
                "username": G.nodes[node]["username"],
                "degree": degree_dict[node],
                "degree_centrality": degree_centrality[node],
                "betweenness_centrality": betweenness.get(node, 0.0),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["degree", "betweenness_centrality"], ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )



def save_degree_histogram(G: nx.Graph, output_path: Path, title: str) -> None:
    degrees = [degree for _, degree in G.degree()]
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=30, edgecolor="black")
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_degree_rank_plot(G: nx.Graph, output_path: Path, title: str) -> None:
    degrees = sorted([degree for _, degree in G.degree()], reverse=True)
    ranks = range(1, len(degrees) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(ranks, degrees, marker="o", linestyle="None", markersize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.title(title)
    plt.xlabel("Rank (log scale)")
    plt.ylabel("Degree (log scale)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_observed_vs_random_plot(observed_df: pd.DataFrame, random_summary_df: pd.DataFrame, output_path: Path, title: str) -> None:
    observed = observed_df.iloc[0]
    random_summary = random_summary_df.iloc[0]
    metrics = [
        ("average_clustering", "Average clustering"),
        ("transitivity", "Transitivity"),
        ("assortativity", "Assortativity"),
        ("average_shortest_path_length_LCC", "Avg shortest path"),
        ("diameter_LCC", "Diameter"),
    ]
    labels = [label for _, label in metrics]
    observed_values = [observed[key] for key, _ in metrics]
    random_values = [random_summary[f"{key}_mean"] for key, _ in metrics]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, observed_values, width, label="Observed")
    plt.bar(x + width / 2, random_values, width, label="Random mean")
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Task C helpers
# ============================================================

def looks_like_bot(username: str) -> bool:
    return "bot" in username.lower()



def choose_suspicious_sources(G: nx.Graph) -> tuple[str, list[str], pd.DataFrame]:
    degree_df = pd.DataFrame(
        [{"username": G.nodes[node]["username"], "degree": degree} for node, degree in G.degree()]
    ).sort_values(["degree", "username"], ascending=[False, True]).reset_index(drop=True)

    human_like = degree_df[~degree_df["username"].apply(looks_like_bot)].copy()
    if len(human_like) > 0:
        single_source = human_like.iloc[0]["username"]
    else:
        single_source = degree_df.iloc[0]["username"]

    single_node = next(node for node in G.nodes() if G.nodes[node]["username"] == single_source)

    neighbour_rows = []
    for neighbour in G.neighbors(single_node):
        username = G.nodes[neighbour]["username"]
        neighbour_rows.append(
            {
                "username": username,
                "degree": G.degree(neighbour),
                "weight_to_single_source": G[single_node][neighbour].get("weight", 1),
                "looks_like_bot": looks_like_bot(username),
            }
        )
    neighbour_df = pd.DataFrame(neighbour_rows)
    if len(neighbour_df) > 0:
        preferred_neighbours = neighbour_df[~neighbour_df["looks_like_bot"]].copy()
        candidate_pool = preferred_neighbours if len(preferred_neighbours) > 0 else neighbour_df
        candidate_pool = candidate_pool.sort_values(
            ["weight_to_single_source", "degree", "username"],
            ascending=[False, False, True],
        )
        second_source = candidate_pool.iloc[0]["username"]
    else:
        fallback_pool = human_like[human_like["username"] != single_source]
        if len(fallback_pool) == 0:
            fallback_pool = degree_df[degree_df["username"] != single_source]
        second_source = fallback_pool.iloc[0]["username"]

    selection_df = pd.DataFrame(
        [
            {
                "selection_rule": "single_source",
                "username": single_source,
                "degree": int(degree_df.loc[degree_df["username"] == single_source, "degree"].iloc[0]),
                "looks_like_bot": looks_like_bot(single_source),
                "notes": "Highest-degree non-bot editor if available; otherwise highest-degree editor overall.",
            },
            {
                "selection_rule": "dual_source_partner",
                "username": second_source,
                "degree": int(degree_df.loc[degree_df["username"] == second_source, "degree"].iloc[0]),
                "looks_like_bot": looks_like_bot(second_source),
                "notes": "Strongest non-bot neighbour of the single-source editor by edge weight, then degree; fallback to next highest-degree editor.",
            },
        ]
    )

    return single_source, [single_source, second_source], selection_df



def edge_weight_by_username(G: nx.Graph, user_a: str, user_b: str, username_to_id: dict[str, int]) -> float:
    a = username_to_id[user_a]
    b = username_to_id[user_b]
    if G.has_edge(a, b):
        return float(G[a][b].get("weight", 1.0))
    return 0.0



def get_distance_maps_by_username(G: nx.Graph, sources: list[str], username_to_id: dict[str, int], cutoff: int = 2) -> dict[str, dict[int, int]]:
    distance_maps = {}
    for username in sources:
        distance_maps[username] = nx.single_source_shortest_path_length(G, username_to_id[username], cutoff=cutoff)
    return distance_maps



def get_candidate_nodes(distance_maps: dict[str, dict[int, int]], source_ids: set[int], max_hops: int = 2) -> set[int]:
    candidates: set[int] = set()
    for dist_map in distance_maps.values():
        for node, distance in dist_map.items():
            if node not in source_ids and 1 <= distance <= max_hops:
                candidates.add(node)
    return candidates



def transmission_probability(weight: float, max_weight_from_source: float, beta: float = 2.0) -> float:
    if weight <= 0 or max_weight_from_source <= 0:
        return 0.0
    norm_weight = weight / max_weight_from_source
    prob = 1 - math.exp(-beta * norm_weight)
    return min(max(prob, 0.0), 1.0)



def safe_product_of_complements(probabilities: Iterable[float]) -> float:
    probabilities = list(probabilities)
    if len(probabilities) == 0:
        return 1.0
    log_sum = 0.0
    for prob in probabilities:
        prob = min(max(prob, 0.0), 1.0)
        if prob >= 1.0:
            return 0.0
        log_sum += math.log1p(-prob)
    return math.exp(log_sum)



def shared_neighbours_score(G: nx.Graph, node: int, sources: list[int]) -> tuple[int, float]:
    node_neighbours = set(G.neighbors(node))
    total_shared = 0
    union_source_neighbours: set[int] = set()
    for source in sources:
        source_neighbours = set(G.neighbors(source))
        total_shared += len(node_neighbours & source_neighbours)
        union_source_neighbours |= source_neighbours
    union_size = len(node_neighbours | union_source_neighbours)
    jaccard_union = len(node_neighbours & union_source_neighbours) / union_size if union_size > 0 else 0.0
    return total_shared, jaccard_union



def normalise_series(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return series
    max_value = series.max()
    min_value = series.min()
    if pd.isna(max_value) or pd.isna(min_value) or max_value == min_value:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_value) / (max_value - min_value)



def scenario_label_from_rank(rank: int) -> str:
    if rank <= 5:
        return "very_high"
    if rank <= 10:
        return "high"
    if rank <= 20:
        return "medium"
    return "lower"



def analyse_propagation_scenario(
    G: nx.Graph,
    sources: list[str],
    username_to_id: dict[str, int],
    top_k: int = 25,
    scenario_name: str = "scenario",
    beta_direct: float = 2.0,
    max_hops: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source_ids = [username_to_id[s] for s in sources]
    distance_maps = get_distance_maps_by_username(G, sources, username_to_id, cutoff=max_hops)
    candidate_nodes = sorted(get_candidate_nodes(distance_maps, set(source_ids), max_hops=max_hops))

    source_stats = {}
    for username, source_id in zip(sources, source_ids):
        incident_weights = [G[source_id][nbr].get("weight", 1.0) for nbr in G.neighbors(source_id)]
        source_stats[username] = {
            "degree": G.degree(source_id),
            "weighted_degree": float(sum(incident_weights)),
            "max_incident_weight": max(incident_weights) if incident_weights else 1.0,
            "neighbours": set(G.neighbors(source_id)),
        }

    one_hop_frontier: set[int] = set()
    for source_id in source_ids:
        one_hop_frontier |= set(G.neighbors(source_id))
    one_hop_frontier -= set(source_ids)

    one_hop_rows = []
    direct_probs = []
    for node in sorted(one_hop_frontier):
        per_source_probs = []
        total_direct_weight = 0.0
        for username in sources:
            weight = edge_weight_by_username(G, username, G.nodes[node]["username"], username_to_id)
            total_direct_weight += weight
            prob = transmission_probability(weight, source_stats[username]["max_incident_weight"], beta=beta_direct)
            per_source_probs.append(prob)
        combined_prob = 1 - safe_product_of_complements(per_source_probs)
        direct_probs.append(combined_prob)
        one_hop_rows.append(
            {
                "editor": G.nodes[node]["username"],
                "direct_sources_connected": sum(1 for prob in per_source_probs if prob > 0),
                "combined_direct_transmission_probability": combined_prob,
                "total_direct_weight_from_sources": total_direct_weight,
                "degree": G.degree(node),
            }
        )
    one_hop_df = (
        pd.DataFrame(one_hop_rows)
        .sort_values(
            ["combined_direct_transmission_probability", "total_direct_weight_from_sources", "degree"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
        if len(one_hop_rows) > 0
        else pd.DataFrame(columns=["editor", "direct_sources_connected", "combined_direct_transmission_probability", "total_direct_weight_from_sources", "degree"])
    )

    no_one_hop_propagation_prob = safe_product_of_complements(direct_probs)
    at_least_one_propagated_prob = 1 - no_one_hop_propagation_prob
    expected_directly_exposed = float(np.sum(direct_probs)) if len(direct_probs) > 0 else 0.0

    rows = []
    for node in candidate_nodes:
        proximity_parts = []
        direct_strength_parts = []
        direct_contact_count = 0
        for username in sources:
            distance = distance_maps[username].get(node, np.inf)
            proximity_parts.append(1 / distance if np.isfinite(distance) and distance > 0 else 0.0)
            source_node = username_to_id[username]
            weight = G[source_node][node].get("weight", 1.0) if G.has_edge(source_node, node) else 0.0
            if weight > 0:
                direct_contact_count += 1
            max_weight = source_stats[username]["max_incident_weight"]
            direct_strength_parts.append(weight / max_weight if max_weight > 0 else 0.0)
        total_shared, jaccard_union = shared_neighbours_score(G, node, source_ids)
        rows.append(
            {
                "editor": G.nodes[node]["username"],
                "degree": G.degree(node),
                "proximity_mean": float(np.mean(proximity_parts)),
                "proximity_min": float(np.min(proximity_parts)),
                "direct_strength_sum": float(np.sum(direct_strength_parts)),
                "direct_contact_fraction": direct_contact_count / len(sources),
                "shared_neighbours_total": total_shared,
                "jaccard_with_source_neighbourhood": jaccard_union,
                "min_distance_to_sources": min(distance_maps[username].get(node, np.inf) for username in sources),
                "distance_map": str({username: distance_maps[username].get(node, None) for username in sources}),
            }
        )

    priority_df = pd.DataFrame(rows)
    if len(priority_df) == 0:
        empty_summary = pd.DataFrame(
            [
                {
                    "scenario_name": scenario_name,
                    "sources": ", ".join(sources),
                    "number_of_sources": len(sources),
                    "source_degrees": str({username: source_stats[username]["degree"] for username in sources}),
                    "source_weighted_degrees": str({username: source_stats[username]["weighted_degree"] for username in sources}),
                    "one_hop_frontier_size": len(one_hop_frontier),
                    "two_hop_candidate_count": 0,
                    "probability_no_one_hop_propagation_yet": no_one_hop_propagation_prob,
                    "probability_at_least_one_one_hop_neighbour_already_exposed": at_least_one_propagated_prob,
                    "expected_directly_exposed_neighbours": expected_directly_exposed,
                    "top_priority_editor": None,
                    "top_priority_score": None,
                }
            ]
        )
        return empty_summary, one_hop_df, priority_df, priority_df

    priority_df["degree_norm"] = normalise_series(np.log1p(priority_df["degree"]))
    priority_df["shared_neighbours_norm"] = normalise_series(priority_df["shared_neighbours_total"])
    priority_df["jaccard_norm"] = normalise_series(priority_df["jaccard_with_source_neighbourhood"])

    if len(sources) == 1:
        priority_df["priority_score"] = (
            0.40 * priority_df["proximity_mean"]
            + 0.25 * priority_df["direct_strength_sum"]
            + 0.20 * priority_df["shared_neighbours_norm"]
            + 0.10 * priority_df["jaccard_norm"]
            + 0.05 * priority_df["degree_norm"]
        )
    else:
        priority_df["priority_score"] = (
            0.25 * priority_df["proximity_mean"]
            + 0.20 * priority_df["proximity_min"]
            + 0.20 * priority_df["direct_strength_sum"]
            + 0.15 * priority_df["direct_contact_fraction"]
            + 0.10 * priority_df["shared_neighbours_norm"]
            + 0.05 * priority_df["jaccard_norm"]
            + 0.05 * priority_df["degree_norm"]
        )

    priority_df = priority_df.sort_values(["priority_score", "direct_contact_fraction", "degree"], ascending=[False, False, False]).reset_index(drop=True)
    priority_df["priority_rank"] = np.arange(1, len(priority_df) + 1)
    priority_df["priority_label"] = priority_df["priority_rank"].apply(scenario_label_from_rank)

    summary_df = pd.DataFrame(
        [
            {
                "scenario_name": scenario_name,
                "sources": ", ".join(sources),
                "number_of_sources": len(sources),
                "source_degrees": str({username: source_stats[username]["degree"] for username in sources}),
                "source_weighted_degrees": str({username: source_stats[username]["weighted_degree"] for username in sources}),
                "one_hop_frontier_size": len(one_hop_frontier),
                "two_hop_candidate_count": len(priority_df),
                "probability_no_one_hop_propagation_yet": no_one_hop_propagation_prob,
                "probability_at_least_one_one_hop_neighbour_already_exposed": at_least_one_propagated_prob,
                "expected_directly_exposed_neighbours": expected_directly_exposed,
                "top_priority_editor": priority_df.iloc[0]["editor"],
                "top_priority_score": priority_df.iloc[0]["priority_score"],
            }
        ]
    )

    return summary_df, one_hop_df, priority_df.head(top_k).reset_index(drop=True), priority_df



def save_priority_scores_plot(priority_df: pd.DataFrame, sources: list[str], output_path: Path, title: str, top_n: int = 15) -> None:
    plot_df = priority_df.head(top_n).copy()
    if len(plot_df) == 0:
        return
    plot_df = plot_df.iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["editor"], plot_df["priority_score"])
    plt.title(title + f"\nSources: {', '.join(sources)}")
    plt.xlabel("Priority score")
    plt.ylabel("Editors")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def save_priority_focus_network(
    G: nx.Graph,
    username_to_id: dict[str, int],
    sources: list[str],
    priority_df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    chosen_nodes = set(username_to_id[s] for s in sources)
    chosen_nodes |= set(username_to_id[username] for username in priority_df.head(top_n)["editor"].tolist())
    H = G.subgraph(chosen_nodes).copy()
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(H, seed=DEFAULT_RANDOM_SEED, k=0.45)
    source_ids = {username_to_id[s] for s in sources}
    top_priority = set(username_to_id[username] for username in priority_df.head(top_n)["editor"].tolist())

    node_sizes = []
    node_colors = []
    labels = {}
    for node in H.nodes():
        if node in source_ids:
            node_sizes.append(800)
            node_colors.append("tab:red")
            labels[node] = H.nodes[node]["username"]
        elif node in top_priority:
            score = float(priority_df.loc[priority_df["editor"] == H.nodes[node]["username"], "priority_score"].iloc[0])
            node_sizes.append(260 + 1000 * score)
            node_colors.append("tab:orange")
            labels[node] = H.nodes[node]["username"]
        else:
            node_sizes.append(180)
            node_colors.append("lightgrey")
    widths = [0.5 + 0.15 * H[u][v].get("weight", 1) for u, v in H.edges()]
    nx.draw_networkx_nodes(H, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85)
    nx.draw_networkx_edges(H, pos, width=widths, alpha=0.35)
    nx.draw_networkx_labels(H, pos, labels=labels, font_size=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
