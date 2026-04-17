from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import pandas as pd

from config_part2 import (
    SELECTED_AREA_NAME,
    BBOX_WGS84,
    BBOX_BNG,
)


OUTPUT_DIR = Path("outputs") / "taskA"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_drive_network():
    """
    Download the drivable road network inside the fixed study-area bbox.
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
    Project the network to British National Grid so all distance-based
    calculations are in metres and consistent with the accident data.
    """
    return ox.projection.project_graph(G, to_crs="EPSG:27700")


def to_undirected_multigraph(G):
    """
    Convert the directed road graph to an undirected multigraph for
    undirected spatial statistics and diameter calculations.
    """
    Gu = nx.MultiGraph()
    Gu.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        Gu.add_edge(u, v, **data)

    return Gu


def largest_connected_component_subgraph(Gu):
    """
    Return the largest connected component of an undirected graph.
    """
    if Gu.number_of_nodes() == 0:
        return Gu.copy()

    largest_cc_nodes = max(nx.connected_components(Gu), key=len)
    return Gu.subgraph(largest_cc_nodes).copy()


def spatial_diameter_m(Gu):
    """
    Compute the weighted spatial diameter in metres using edge length.
    If the full graph is disconnected, compute it on the largest connected component.
    """
    H = largest_connected_component_subgraph(Gu)

    if H.number_of_nodes() <= 1:
        return 0.0, 0

    max_distance = 0.0
    for _, lengths in nx.all_pairs_dijkstra_path_length(H, weight="length"):
        local_max = max(lengths.values())
        if local_max > max_distance:
            max_distance = local_max

    return float(max_distance), H.number_of_nodes()


def check_topological_planarity(G):
    """
    Check topological planarity on a simple undirected version of the graph.
    """
    G_simple = nx.Graph()
    G_simple.add_nodes_from(G.nodes())

    for u, v in G.edges():
        G_simple.add_edge(u, v)

    is_planar, _ = nx.check_planarity(G_simple, counterexample=False)
    return is_planar


def find_geometric_crossing_examples(edges_gdf, max_examples=5):
    """
    Find a few candidate geometric crossing examples where two road geometries
    cross but do not share endpoints. These are useful for discussing
    non-planarity caused by overpasses or grade-separated roads.
    """
    edges = edges_gdf.reset_index()[["u", "v", "key", "geometry", "length"]].copy()
    edges = edges[edges["geometry"].notna()].reset_index(drop=True)

    if edges.empty:
        return pd.DataFrame()

    sindex = edges.sindex
    examples = []

    for i, row_i in edges.iterrows():
        candidate_idx = list(sindex.intersection(row_i.geometry.bounds))

        for j in candidate_idx:
            if j <= i:
                continue

            row_j = edges.iloc[j]

            # Skip edges sharing an endpoint
            if len({row_i.u, row_i.v, row_j.u, row_j.v}) < 4:
                continue

            geom_i = row_i.geometry
            geom_j = row_j.geometry

            if not geom_i.crosses(geom_j):
                continue

            inter = geom_i.intersection(geom_j)

            if inter.is_empty:
                continue

            if inter.geom_type == "Point":
                examples.append(
                    {
                        "edge1_u": row_i.u,
                        "edge1_v": row_i.v,
                        "edge1_key": row_i.key,
                        "edge2_u": row_j.u,
                        "edge2_v": row_j.v,
                        "edge2_key": row_j.key,
                        "crossing_x": inter.x,
                        "crossing_y": inter.y,
                    }
                )

            elif inter.geom_type == "MultiPoint":
                for pt in inter.geoms:
                    examples.append(
                        {
                            "edge1_u": row_i.u,
                            "edge1_v": row_i.v,
                            "edge1_key": row_i.key,
                            "edge2_u": row_j.u,
                            "edge2_v": row_j.v,
                            "edge2_key": row_j.key,
                            "crossing_x": pt.x,
                            "crossing_y": pt.y,
                        }
                    )

            if len(examples) >= max_examples:
                return pd.DataFrame(examples)

    return pd.DataFrame(examples)


def compute_task_a_metrics(G_proj):
    """
    Compute the main Task A metrics.
    """
    area_m2 = (BBOX_BNG["east"] - BBOX_BNG["west"]) * (BBOX_BNG["north"] - BBOX_BNG["south"])

    basic = ox.stats.basic_stats(G_proj, area=area_m2)

    Gu = to_undirected_multigraph(G_proj)
    diameter_m, lcc_nodes = spatial_diameter_m(Gu)

    metrics = {
        "selected_area_name": SELECTED_AREA_NAME,
        "bbox_west": BBOX_WGS84["west"],
        "bbox_south": BBOX_WGS84["south"],
        "bbox_east": BBOX_WGS84["east"],
        "bbox_north": BBOX_WGS84["north"],
        "area_sq_km": area_m2 / 1_000_000,
        "nodes_n": basic["n"],
        "edges_m": basic["m"],
        "street_segment_count": basic["street_segment_count"],
        "spatial_diameter_m_lcc": diameter_m,
        "spatial_diameter_km_lcc": diameter_m / 1000,
        "largest_connected_component_nodes": lcc_nodes,
        "average_street_length_m": basic["street_length_avg"],
        "node_density_per_sq_km": basic["node_density_km"],
        "intersection_count": basic["intersection_count"],
        "intersection_density_per_sq_km": basic["intersection_density_km"],
        "edge_density_km_per_sq_km": basic["edge_density_km"],
        "street_density_km_per_sq_km": basic["street_density_km"],
        "average_circuity": basic["circuity_avg"],
        "average_node_degree_k_avg": basic["k_avg"],
        "streets_per_node_avg": basic["streets_per_node_avg"],
    }

    return pd.DataFrame([metrics])


def add_street_count_if_missing(G_proj, nodes_gdf):
    """
    Ensure nodes have street_count for plotting intersections.
    """
    if "street_count" not in nodes_gdf.columns:
        street_counts = ox.stats.count_streets_per_node(G_proj)
        nodes_gdf = nodes_gdf.copy()
        nodes_gdf["street_count"] = nodes_gdf.index.map(street_counts)

    return nodes_gdf


def plot_network_overview(nodes_gdf, edges_gdf, save_path):
    """
    Plot the projected road network.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    edges_gdf.plot(ax=ax, linewidth=0.8)
    nodes_gdf.plot(ax=ax, markersize=4)
    ax.set_title("Leeds central road network (drive)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_intersections(nodes_gdf, edges_gdf, save_path):
    """
    Plot the road network and highlight intersections.
    """
    intersection_nodes = nodes_gdf[nodes_gdf["street_count"] >= 2].copy()

    fig, ax = plt.subplots(figsize=(8, 8))
    edges_gdf.plot(ax=ax, linewidth=0.8)
    if not intersection_nodes.empty:
        intersection_nodes.plot(ax=ax, markersize=10)
    ax.set_title("Leeds central road network with intersections highlighted")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_outputs(metrics_df, nodes_gdf, edges_gdf, crossing_examples_df):
    """
    Save Task A tables.
    """
    metrics_df.to_csv(OUTPUT_DIR / "taskA_metrics.csv", index=False)
    nodes_gdf.reset_index().to_csv(OUTPUT_DIR / "taskA_nodes.csv", index=False)
    edges_gdf.reset_index().to_csv(OUTPUT_DIR / "taskA_edges.csv", index=False)

    if crossing_examples_df.empty:
        pd.DataFrame(columns=[
            "edge1_u", "edge1_v", "edge1_key",
            "edge2_u", "edge2_v", "edge2_key",
            "crossing_x", "crossing_y"
        ]).to_csv(OUTPUT_DIR / "taskA_crossing_examples.csv", index=False)
    else:
        crossing_examples_df.to_csv(OUTPUT_DIR / "taskA_crossing_examples.csv", index=False)


def main():
    print("=" * 78)
    print("Task A - Leeds road network construction and spatial metrics")
    print("=" * 78)

    G = download_drive_network()
    G_proj = project_network_to_bng(G)

    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(G_proj)
    nodes_gdf = add_street_count_if_missing(G_proj, nodes_gdf)

    metrics_df = compute_task_a_metrics(G_proj)

    topological_planarity = check_topological_planarity(G_proj)
    metrics_df["topological_planarity"] = topological_planarity

    crossing_examples_df = find_geometric_crossing_examples(edges_gdf, max_examples=5)
    metrics_df["geometric_crossing_examples_found"] = len(crossing_examples_df)

    save_outputs(metrics_df, nodes_gdf, edges_gdf, crossing_examples_df)

    plot_network_overview(
        nodes_gdf,
        edges_gdf,
        OUTPUT_DIR / "taskA_network_overview.png",
    )

    plot_intersections(
        nodes_gdf,
        edges_gdf,
        OUTPUT_DIR / "taskA_intersections.png",
    )

    print("\nMain Task A metrics:")
    print(metrics_df.to_string(index=False))

    if crossing_examples_df.empty:
        print("\nNo geometric crossing examples were detected by the crossing search.")
    else:
        print("\nSample geometric crossing examples:")
        print(crossing_examples_df.to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()