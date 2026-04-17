from pathlib import Path
from itertools import combinations

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from config_part2 import (
    BBOX_WGS84,
    PROCESSED_DIR,
    SELECTED_AREA_NAME,
)


OUTPUT_DIR = Path("outputs") / "taskC"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MARATHON_M = 42000
ROUTE_TOLERANCE_M = 500
N_SEEDS = 4


def download_drive_network():
    """
    Download the drivable road network for the fixed study area.
    """
    bbox = (
        BBOX_WGS84["west"],
        BBOX_WGS84["south"],
        BBOX_WGS84["east"],
        BBOX_WGS84["north"],
    )

    G = ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type="drive",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )
    return G


def project_network_to_bng(G):
    """
    Project the network to British National Grid.
    """
    return ox.projection.project_graph(G, to_crs="EPSG:27700")


def get_network_gdfs(G_proj):
    """
    Convert the graph to GeoDataFrames and remove duplicate opposite-direction geometries.
    """
    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(G_proj)

    nodes_gdf = nodes_gdf.copy()
    edges_gdf = edges_gdf.reset_index().copy()

    street_counts = ox.stats.count_streets_per_node(G_proj)
    nodes_gdf["street_count"] = nodes_gdf.index.map(street_counts)

    edges_gdf["geom_wkb"] = edges_gdf["geometry"].apply(lambda g: g.wkb)
    edges_gdf = (
        edges_gdf.sort_values("length")
        .drop_duplicates(subset=["geom_wkb"])
        .drop(columns=["geom_wkb"])
        .reset_index(drop=True)
    )

    edges_gdf["edge_id"] = np.arange(len(edges_gdf))
    return nodes_gdf, edges_gdf


def build_simple_undirected_graph(nodes_gdf, edges_gdf):
    """
    Build a simple undirected weighted graph from the projected network.
    """
    G_simple = nx.Graph()

    for node_id, row in nodes_gdf.iterrows():
        G_simple.add_node(
            node_id,
            x=row.geometry.x,
            y=row.geometry.y,
            street_count=row["street_count"],
        )

    for row in edges_gdf.itertuples(index=False):
        u = row.u
        v = row.v
        length = float(row.length)

        if G_simple.has_edge(u, v):
            if length < G_simple[u][v]["length"]:
                G_simple[u][v]["length"] = length
                G_simple[u][v]["geometry"] = row.geometry
                G_simple[u][v]["edge_id"] = row.edge_id
        else:
            G_simple.add_edge(
                u,
                v,
                length=length,
                geometry=row.geometry,
                edge_id=row.edge_id,
            )

    return G_simple


def load_accidents_in_selected_area():
    """
    Load the preprocessed accidents inside the selected area.
    """
    csv_path = PROCESSED_DIR / "accidents_in_selected_area_2012_2016.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot find {csv_path}. Run preprocess_accidents_part2.py first."
        )

    df = pd.read_csv(csv_path)
    required = ["accident_uid", "year", "easting", "northing"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Accident file is missing required columns: {missing}")

    gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["easting"], df["northing"]),
        crs="EPSG:27700",
    )

    return gdf


def snap_accidents_to_nearest_edges(accidents_gdf, edges_gdf):
    """
    Attach each accident to its nearest road segment.
    """
    edge_join = edges_gdf[["edge_id", "u", "v", "length", "geometry"]].copy()

    snapped = gpd.sjoin_nearest(
        accidents_gdf,
        edge_join,
        how="left",
        distance_col="snap_distance_m",
        lsuffix="acc",
        rsuffix="edge",
    )

    snapped = (
        snapped.sort_values(["accident_uid", "snap_distance_m"])
        .drop_duplicates(subset=["accident_uid"], keep="first")
        .reset_index(drop=True)
    )

    return snapped


def compute_edge_accident_counts(edges_gdf, snapped_accidents_gdf):
    """
    Count how many accidents are assigned to each road segment.
    """
    counts = snapped_accidents_gdf.groupby("edge_id").size()
    edges_with_counts = edges_gdf.copy()
    edges_with_counts["accident_count"] = (
        edges_with_counts["edge_id"].map(counts).fillna(0).astype(int)
    )
    return edges_with_counts


def build_node_candidate_table(G_simple, nodes_gdf, edges_with_counts_gdf):
    """
    Build candidate seed nodes using intersections in the largest connected component.
    """
    largest_cc_nodes = max(nx.connected_components(G_simple), key=len)
    candidate_nodes = nodes_gdf.loc[list(largest_cc_nodes)].copy()

    candidate_nodes = candidate_nodes[candidate_nodes["street_count"] >= 3].copy()
    if candidate_nodes.empty:
        candidate_nodes = nodes_gdf.loc[list(largest_cc_nodes)].copy()
        candidate_nodes = candidate_nodes[candidate_nodes["street_count"] >= 2].copy()

    edge_accident_map = edges_with_counts_gdf.set_index("edge_id")["accident_count"].to_dict()

    incident_accident_counts = {}
    incident_road_lengths = {}
    graph_degrees = {}

    for node_id in candidate_nodes.index:
        graph_degrees[node_id] = G_simple.degree(node_id)

        total_accidents = 0
        total_length = 0.0
        for nbr in G_simple.neighbors(node_id):
            edge_id = G_simple[node_id][nbr]["edge_id"]
            total_accidents += edge_accident_map.get(edge_id, 0)
            total_length += G_simple[node_id][nbr]["length"]

        incident_accident_counts[node_id] = total_accidents
        incident_road_lengths[node_id] = total_length

    candidate_nodes["graph_degree"] = candidate_nodes.index.map(graph_degrees)
    candidate_nodes["incident_accident_count"] = candidate_nodes.index.map(incident_accident_counts)
    candidate_nodes["incident_road_length_m"] = candidate_nodes.index.map(incident_road_lengths)
    candidate_nodes["x"] = candidate_nodes.geometry.x
    candidate_nodes["y"] = candidate_nodes.geometry.y

    return candidate_nodes


def normalise_series(series):
    if series.max() == series.min():
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def euclidean_distance(row_a, row_b):
    return float(np.hypot(row_a["x"] - row_b["x"], row_a["y"] - row_b["y"]))


def select_seed_nodes(candidate_nodes, n_seeds=4, strategy="initial"):
    """
    Select N seed nodes for the Voronoi partition.

    initial:
        stronger emphasis on low accident exposure and spatial spread

    revised:
        stronger emphasis on spread and local connectivity
    """
    df = candidate_nodes.copy()

    df["accident_norm"] = normalise_series(df["incident_accident_count"])
    connectivity_raw = 0.5 * normalise_series(df["street_count"]) + 0.5 * normalise_series(df["incident_road_length_m"])
    df["connectivity_norm"] = connectivity_raw

    if strategy == "initial":
        df = df.sort_values(
            by=["incident_accident_count", "street_count", "incident_road_length_m"],
            ascending=[True, False, False],
        ).copy()
    else:
        df = df.sort_values(
            by=["street_count", "incident_road_length_m", "incident_accident_count"],
            ascending=[False, False, True],
        ).copy()

    selected = [df.index[0]]

    while len(selected) < n_seeds:
        remaining = df.loc[~df.index.isin(selected)].copy()

        min_distances = []
        for node_id, row in remaining.iterrows():
            dists = [euclidean_distance(row, df.loc[s]) for s in selected]
            min_distances.append(min(dists))
        remaining["min_distance_to_selected"] = min_distances
        remaining["distance_norm"] = normalise_series(remaining["min_distance_to_selected"])

        if strategy == "initial":
            remaining["selection_score"] = (
                0.60 * remaining["distance_norm"]
                - 0.30 * remaining["accident_norm"]
                + 0.10 * remaining["connectivity_norm"]
            )
        else:
            remaining["selection_score"] = (
                0.50 * remaining["distance_norm"]
                + 0.35 * remaining["connectivity_norm"]
                - 0.15 * remaining["accident_norm"]
            )

        next_seed = remaining.sort_values(
            by=["selection_score", "street_count", "incident_road_length_m"],
            ascending=[False, False, False],
        ).index[0]

        selected.append(next_seed)

    seed_table = df.loc[selected].copy()
    seed_table["seed_order"] = range(1, len(seed_table) + 1)
    seed_table["strategy"] = strategy
    return selected, seed_table


def compute_seed_distance_maps(G_simple, seed_nodes):
    """
    Compute shortest-path lengths from each seed to all nodes.
    """
    distance_maps = {}
    for seed in seed_nodes:
        distance_maps[seed] = nx.single_source_dijkstra_path_length(
            G_simple,
            seed,
            weight="length",
        )
    return distance_maps


def assign_nodes_to_cells(G_simple, seed_nodes, distance_maps):
    """
    Assign each node to the closest seed by weighted shortest-path distance.
    """
    node_assignment = {}

    for node in G_simple.nodes():
        best_seed = None
        best_distance = np.inf

        for seed in seed_nodes:
            d = distance_maps[seed].get(node, np.inf)
            if d < best_distance:
                best_distance = d
                best_seed = seed

        node_assignment[node] = best_seed

    return node_assignment


def assign_edges_to_cells(edges_gdf, seed_nodes, distance_maps):
    """
    Assign each edge to the seed with the smallest average endpoint distance.
    """
    assignments = []

    for row in edges_gdf.itertuples(index=False):
        best_seed = None
        best_distance = np.inf

        for seed in seed_nodes:
            du = distance_maps[seed].get(row.u, np.inf)
            dv = distance_maps[seed].get(row.v, np.inf)
            avg_d = (du + dv) / 2

            if avg_d < best_distance:
                best_distance = avg_d
                best_seed = seed

        assignments.append(best_seed)

    result = edges_gdf.copy()
    result["cell_seed"] = assignments
    return result


def build_cell_subgraph(G_simple, node_assignment, seed):
    """
    Build the subgraph induced by nodes assigned to one seed.
    """
    nodes = [n for n, s in node_assignment.items() if s == seed]
    H = G_simple.subgraph(nodes).copy()

    if H.number_of_nodes() == 0:
        return H

    largest_cc_nodes = max(nx.connected_components(H), key=len)
    return H.subgraph(largest_cc_nodes).copy()


def cycle_length_m(H, cycle_nodes):
    """
    Compute the total length of a simple cycle.
    """
    total = 0.0
    closed = cycle_nodes + [cycle_nodes[0]]
    for a, b in zip(closed[:-1], closed[1:]):
        total += H[a][b]["length"]
    return float(total)


def rotate_cycle_to_anchor(cycle_nodes, anchor):
    """
    Rotate a cycle so that it starts and ends at the chosen anchor node.
    """
    idx = cycle_nodes.index(anchor)
    rotated = cycle_nodes[idx:] + cycle_nodes[:idx]
    return rotated + [rotated[0]]


def extract_cycle_catalog(H):
    """
    Extract a catalog of candidate base loops from the cycle basis.
    """
    if H.number_of_nodes() == 0 or H.number_of_edges() == 0:
        return []

    cycles = nx.cycle_basis(H)
    catalog = []

    for i, cyc in enumerate(cycles):
        if len(cyc) < 3:
            continue

        length_m = cycle_length_m(H, cyc)

        if length_m <= 0:
            continue

        catalog.append(
            {
                "cycle_id": i,
                "nodes": cyc,
                "node_set": set(cyc),
                "length_m": length_m,
            }
        )

    catalog = sorted(catalog, key=lambda x: x["length_m"], reverse=True)
    return catalog


def find_best_closed_route(H, target_m=42000, tolerance_m=500):
    """
    Find an approximately 42 km closed route by repeating one simple base loop.

    The route is selected by:
    1) minimal difference from the target distance
    2) maximal base-loop length to avoid excessive repetition
    """
    catalog = extract_cycle_catalog(H)

    if not catalog:
        return None

    best = None

    for item in catalog:
        loop_length = item["length_m"]
        if loop_length <= 0:
            continue

        max_reps = int(np.ceil((target_m + tolerance_m) / loop_length)) + 1

        for reps in range(1, max_reps + 1):
            total_length = reps * loop_length
            diff = abs(total_length - target_m)

            if diff <= tolerance_m:
                candidate = {
                    "route_type": "single_loop_repeated",
                    "base_cycle_nodes": item["nodes"],
                    "base_loop_length_m": loop_length,
                    "repetitions": reps,
                    "total_route_length_m": total_length,
                    "difference_from_target_m": diff,
                }

                if best is None:
                    best = candidate
                else:
                    if candidate["difference_from_target_m"] < best["difference_from_target_m"]:
                        best = candidate
                    elif (
                        candidate["difference_from_target_m"] == best["difference_from_target_m"]
                        and candidate["base_loop_length_m"] > best["base_loop_length_m"]
                    ):
                        best = candidate

    return best


def build_route_edge_gdf(H, route_info, seed):
    """
    Build a GeoDataFrame of the base loop used for plotting.
    """
    base_nodes = route_info["base_cycle_nodes"]
    anchor = seed if seed in base_nodes else base_nodes[0]
    closed_nodes = rotate_cycle_to_anchor(base_nodes, anchor)

    rows = []
    for a, b in zip(closed_nodes[:-1], closed_nodes[1:]):
        rows.append(
            {
                "u": a,
                "v": b,
                "length": H[a][b]["length"],
                "geometry": H[a][b]["geometry"],
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:27700")


def plot_voronoi_cells(edges_with_cell_gdf, nodes_gdf, seed_table, save_path, title):
    """
    Plot the node-network Voronoi partition on the road network.
    """
    color_list = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    seed_to_color = {
        seed: color_list[i % len(color_list)]
        for i, seed in enumerate(seed_table.index.tolist())
    }

    fig, ax = plt.subplots(figsize=(8, 8))

    for seed, group in edges_with_cell_gdf.groupby("cell_seed"):
        color = seed_to_color.get(seed, "lightgray")
        group.plot(ax=ax, linewidth=1.8, color=color, alpha=0.85)

    seed_points = nodes_gdf.loc[seed_table.index].copy()
    seed_points["plot_color"] = [seed_to_color[s] for s in seed_points.index]

    for node_id, row in seed_points.iterrows():
        ax.scatter(row.geometry.x, row.geometry.y, s=60, marker="o")
        ax.text(row.geometry.x, row.geometry.y, str(seed_table.loc[node_id, "seed_order"]), fontsize=9)

    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_route_in_cell(cell_edges_gdf, route_edges_gdf, seed_point, save_path, title):
    """
    Plot a candidate marathon route inside one Voronoi cell.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    cell_edges_gdf.plot(ax=ax, linewidth=1.1, color="lightgray")
    route_edges_gdf.plot(ax=ax, linewidth=2.8, color="tab:red")
    ax.scatter(seed_point.x, seed_point.y, s=80, marker="o")
    ax.set_title(title)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def summarise_cells(seed_nodes, node_assignment, edges_with_cells_gdf, G_simple):
    """
    Summarise each Voronoi cell.
    """
    rows = []

    for seed in seed_nodes:
        cell_nodes = [n for n, s in node_assignment.items() if s == seed]
        cell_edges = edges_with_cells_gdf[edges_with_cells_gdf["cell_seed"] == seed].copy()
        H = build_cell_subgraph(G_simple, node_assignment, seed)

        rows.append(
            {
                "seed_node": seed,
                "cell_nodes": len(cell_nodes),
                "cell_edges": len(cell_edges),
                "cell_total_edge_length_km": float(cell_edges["length"].sum() / 1000),
                "largest_connected_component_nodes": H.number_of_nodes(),
                "largest_connected_component_edges": H.number_of_edges(),
            }
        )

    return pd.DataFrame(rows)


def run_voronoi_strategy(strategy_name, G_simple, nodes_gdf, edges_gdf, candidate_nodes):
    """
    Run one full Voronoi strategy:
    - choose seeds
    - build node-network Voronoi cells
    - summarise cells
    - search for approximate 42 km closed routes
    """
    seed_nodes, seed_table = select_seed_nodes(
        candidate_nodes,
        n_seeds=N_SEEDS,
        strategy=strategy_name,
    )

    distance_maps = compute_seed_distance_maps(G_simple, seed_nodes)
    node_assignment = assign_nodes_to_cells(G_simple, seed_nodes, distance_maps)
    edges_with_cells_gdf = assign_edges_to_cells(edges_gdf, seed_nodes, distance_maps)

    cell_summary_df = summarise_cells(seed_nodes, node_assignment, edges_with_cells_gdf, G_simple)

    route_rows = []
    successful_route_specs = {}

    for seed in seed_nodes:
        H = build_cell_subgraph(G_simple, node_assignment, seed)
        route_info = find_best_closed_route(
            H,
            target_m=TARGET_MARATHON_M,
            tolerance_m=ROUTE_TOLERANCE_M,
        )

        if route_info is None:
            route_rows.append(
                {
                    "strategy": strategy_name,
                    "seed_node": seed,
                    "route_found": False,
                    "route_type": None,
                    "base_loop_length_m": np.nan,
                    "repetitions": np.nan,
                    "total_route_length_m": np.nan,
                    "difference_from_target_m": np.nan,
                }
            )
        else:
            route_rows.append(
                {
                    "strategy": strategy_name,
                    "seed_node": seed,
                    "route_found": True,
                    "route_type": route_info["route_type"],
                    "base_loop_length_m": route_info["base_loop_length_m"],
                    "repetitions": route_info["repetitions"],
                    "total_route_length_m": route_info["total_route_length_m"],
                    "difference_from_target_m": route_info["difference_from_target_m"],
                }
            )
            successful_route_specs[seed] = (H, route_info)

    route_summary_df = pd.DataFrame(route_rows)

    return {
        "strategy_name": strategy_name,
        "seed_nodes": seed_nodes,
        "seed_table": seed_table,
        "distance_maps": distance_maps,
        "node_assignment": node_assignment,
        "edges_with_cells_gdf": edges_with_cells_gdf,
        "cell_summary_df": cell_summary_df,
        "route_summary_df": route_summary_df,
        "successful_route_specs": successful_route_specs,
    }


def save_gdf_csv(gdf, path):
    """
    Save a GeoDataFrame to CSV with geometry converted to WKT.
    """
    df = gdf.copy()
    if "geometry" in df.columns:
        df["geometry_wkt"] = df.geometry.to_wkt()
        df = df.drop(columns="geometry")
    df.to_csv(path, index=False)


def save_strategy_outputs(result, nodes_gdf):
    """
    Save tables and figures for one strategy.
    """
    strategy_name = result["strategy_name"]

    result["seed_table"].to_csv(
        OUTPUT_DIR / f"taskC_{strategy_name}_seeds.csv",
        index=True,
    )
    result["cell_summary_df"].to_csv(
        OUTPUT_DIR / f"taskC_{strategy_name}_cell_summary.csv",
        index=False,
    )
    result["route_summary_df"].to_csv(
        OUTPUT_DIR / f"taskC_{strategy_name}_route_summary.csv",
        index=False,
    )
    save_gdf_csv(
        result["edges_with_cells_gdf"],
        OUTPUT_DIR / f"taskC_{strategy_name}_edges_with_cells.csv",
    )

    plot_voronoi_cells(
        result["edges_with_cells_gdf"],
        nodes_gdf,
        result["seed_table"],
        OUTPUT_DIR / f"taskC_{strategy_name}_voronoi.png",
        title=f"Node-network Voronoi cells ({strategy_name})",
    )

    successful = result["route_summary_df"][result["route_summary_df"]["route_found"]].copy()

    # Plot up to 3 successful cells for the initial strategy
    # and all successful cells for the revised strategy
    if strategy_name == "initial":
        seeds_to_plot = successful["seed_node"].head(3).tolist()
    else:
        seeds_to_plot = successful["seed_node"].tolist()

    for seed in seeds_to_plot:
        H, route_info = result["successful_route_specs"][seed]
        route_edges_gdf = build_route_edge_gdf(H, route_info, seed)
        cell_edges_gdf = result["edges_with_cells_gdf"][
            result["edges_with_cells_gdf"]["cell_seed"] == seed
        ].copy()
        seed_point = nodes_gdf.loc[seed].geometry

        plot_route_in_cell(
            cell_edges_gdf,
            route_edges_gdf,
            seed_point,
            OUTPUT_DIR / f"taskC_{strategy_name}_route_seed_{seed}.png",
            title=(
                f"{strategy_name.capitalize()} strategy - seed {seed} - "
                f"{route_info['total_route_length_m'] / 1000:.2f} km closed route"
            ),
        )


def main():
    print("=" * 78)
    print("Task C - Voronoi cells and marathon route search")
    print("=" * 78)

    G = download_drive_network()
    G_proj = project_network_to_bng(G)
    nodes_gdf, edges_gdf = get_network_gdfs(G_proj)
    G_simple = build_simple_undirected_graph(nodes_gdf, edges_gdf)

    accidents_gdf = load_accidents_in_selected_area()
    snapped_accidents_gdf = snap_accidents_to_nearest_edges(accidents_gdf, edges_gdf)
    edges_with_counts_gdf = compute_edge_accident_counts(edges_gdf, snapped_accidents_gdf)

    candidate_nodes = build_node_candidate_table(G_simple, nodes_gdf, edges_with_counts_gdf)

    initial_result = run_voronoi_strategy(
        "initial",
        G_simple,
        nodes_gdf,
        edges_gdf,
        candidate_nodes,
    )

    revised_result = run_voronoi_strategy(
        "revised",
        G_simple,
        nodes_gdf,
        edges_gdf,
        candidate_nodes,
    )

    save_strategy_outputs(initial_result, nodes_gdf)
    save_strategy_outputs(revised_result, nodes_gdf)

    initial_success = int(initial_result["route_summary_df"]["route_found"].sum())
    revised_success = int(revised_result["route_summary_df"]["route_found"].sum())

    comparison_df = pd.DataFrame(
        [
            {
                "selected_area_name": SELECTED_AREA_NAME,
                "n_cells": N_SEEDS,
                "target_route_length_km": TARGET_MARATHON_M / 1000,
                "route_tolerance_m": ROUTE_TOLERANCE_M,
                "initial_cells_with_route": initial_success,
                "revised_cells_with_route": revised_success,
            }
        ]
    )
    comparison_df.to_csv(OUTPUT_DIR / "taskC_strategy_comparison.csv", index=False)

    print("\nInitial strategy seeds:")
    print(initial_result["seed_table"][[
        "seed_order",
        "street_count",
        "incident_accident_count",
        "incident_road_length_m",
    ]].to_string())

    print("\nInitial strategy route summary:")
    print(initial_result["route_summary_df"].to_string(index=False))

    print("\nRevised strategy seeds:")
    print(revised_result["seed_table"][[
        "seed_order",
        "street_count",
        "incident_accident_count",
        "incident_road_length_m",
    ]].to_string())

    print("\nRevised strategy route summary:")
    print(revised_result["route_summary_df"].to_string(index=False))

    print("\nStrategy comparison:")
    print(comparison_df.to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()