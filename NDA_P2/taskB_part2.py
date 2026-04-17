from pathlib import Path
from itertools import combinations

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import spaghetti
from esda import Moran
from libpysal.weights import W

from config_part2 import (
    BBOX_WGS84,
    SELECTED_AREA_NAME,
    PROCESSED_DIR,
)


OUTPUT_DIR = Path("outputs") / "taskB"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    Convert the graph to GeoDataFrames and remove duplicate physical edges.
    """
    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(G_proj)

    nodes_gdf = nodes_gdf.copy()
    edges_gdf = edges_gdf.reset_index().copy()

    street_counts = ox.stats.count_streets_per_node(G_proj)
    nodes_gdf["street_count"] = nodes_gdf.index.map(street_counts)

    # Remove duplicate opposite-direction geometries
    edges_gdf["geom_wkb"] = edges_gdf["geometry"].apply(lambda g: g.wkb)
    edges_gdf = (
        edges_gdf.sort_values("length")
        .drop_duplicates(subset=["geom_wkb"])
        .drop(columns=["geom_wkb"])
        .reset_index(drop=True)
    )

    edges_gdf["edge_id"] = np.arange(len(edges_gdf))
    return nodes_gdf, edges_gdf


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


def plot_accident_distribution(edges_gdf, accidents_gdf, save_path):
    """
    Plot accident points over the road network.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    edges_gdf.plot(ax=ax, linewidth=0.8, alpha=0.8)
    accidents_gdf.plot(ax=ax, markersize=8, alpha=0.6)
    ax.set_title("Road accidents on the Leeds central road network (2012-2016)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


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

    # Keep one match per accident if ties occur
    snapped = (
        snapped.sort_values(["accident_uid", "snap_distance_m"])
        .drop_duplicates(subset=["accident_uid"], keep="first")
        .reset_index(drop=True)
    )

    return snapped


def build_edge_adjacency_weights(edges_gdf):
    """
    Build a binary weights object where two road segments are neighbors
    if they share an endpoint.
    """
    node_to_edges = {}

    for row in edges_gdf.itertuples(index=False):
        node_to_edges.setdefault(row.u, []).append(row.edge_id)
        node_to_edges.setdefault(row.v, []).append(row.edge_id)

    edge_ids = edges_gdf["edge_id"].tolist()
    neighbors = {edge_id: set() for edge_id in edge_ids}

    for _, incident_edges in node_to_edges.items():
        if len(incident_edges) < 2:
            continue
        for a, b in combinations(incident_edges, 2):
            neighbors[a].add(b)
            neighbors[b].add(a)

    neighbors = {k: sorted(v) for k, v in neighbors.items()}
    w = W(neighbors, silence_warnings=True)
    return w


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


def compute_morans_i(edges_with_counts_gdf):
    """
    Compute Moran's I for accident counts across adjacent road segments.
    """
    edges_sorted = edges_with_counts_gdf.sort_values("edge_id").reset_index(drop=True)
    w = build_edge_adjacency_weights(edges_sorted)
    y = edges_sorted["accident_count"].to_numpy()
    moran = Moran(y, w, permutations=999)
    return moran


def plot_edge_accident_counts(edges_with_counts_gdf, save_path):
    """
    Plot accident counts by road segment.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    edges_with_counts_gdf.plot(
        ax=ax,
        column="accident_count",
        linewidth=2.0,
        legend=True,
    )
    ax.set_title("Accident counts by nearest road segment")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_network_k_function(edges_gdf, accidents_gdf):
    """
    Compute the network-constrained global auto K-function using spaghetti.
    """
    network_lines = edges_gdf[["geometry"]].copy()
    point_data = accidents_gdf[["geometry"]].copy()

    ntw = spaghetti.Network(in_data=network_lines)
    ntw.snapobservations(point_data, "accidents", attribute=False)

    point_pattern = ntw.pointpatterns["accidents"]
    k_result = ntw.GlobalAutoK(
        point_pattern,
        nsteps=20,
        permutations=99,
    )

    return ntw, k_result


def save_k_function_table(k_result, save_path):
    """
    Save K-function results as a table.
    """
    k_df = pd.DataFrame(
        {
            "distance_r": k_result.xaxis,
            "observed_k": k_result.observed,
            "upper_envelope": k_result.upperenvelope,
            "lower_envelope": k_result.lowerenvelope,
        }
    )
    k_df["observed_above_upper"] = k_df["observed_k"] > k_df["upper_envelope"]
    k_df["observed_below_lower"] = k_df["observed_k"] < k_df["lower_envelope"]
    k_df.to_csv(save_path, index=False)
    return k_df


def plot_k_function(k_df, save_path):
    """
    Plot the K-function with simulation envelopes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_df["distance_r"], k_df["observed_k"], linewidth=1.8, label="Observed")
    ax.plot(k_df["distance_r"], k_df["upper_envelope"], linestyle="--", label="Upper envelope")
    ax.plot(k_df["distance_r"], k_df["lower_envelope"], linestyle="--", label="Lower envelope")
    ax.set_xlabel("Network distance r (m)")
    ax.set_ylabel("K(r)")
    ax.set_title("Network-constrained K-function")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_fraction_to_nearest_intersection(snapped_accidents_gdf, nodes_gdf):
    """
    For each accident, compute the fraction of the assigned road-segment length
    from the accident's snapped position to the nearest intersection endpoint.

    Intersections are defined as nodes with street_count >= 2.
    """
    intersection_nodes = set(nodes_gdf[nodes_gdf["street_count"] >= 2].index)

    rows = []
    for row in snapped_accidents_gdf.itertuples(index=False):
        point = row.geometry
        line = row.edge_geometry

        if line is None or getattr(line, "is_empty", False) or line.length == 0:
            frac = np.nan
            dist_to_nearest_intersection = np.nan
            edge_length = np.nan
        else:
            projected_distance = line.project(point)
            dist_to_start = projected_distance
            dist_to_end = line.length - projected_distance

            start_is_intersection = row.edge_u in intersection_nodes
            end_is_intersection = row.edge_v in intersection_nodes

            if start_is_intersection and end_is_intersection:
                dist_to_nearest_intersection = min(dist_to_start, dist_to_end)
            elif start_is_intersection:
                dist_to_nearest_intersection = dist_to_start
            elif end_is_intersection:
                dist_to_nearest_intersection = dist_to_end
            else:
                dist_to_nearest_intersection = np.nan

            frac = (
                dist_to_nearest_intersection / line.length
                if pd.notna(dist_to_nearest_intersection)
                else np.nan
            )
            edge_length = line.length

        rows.append(
            {
                "accident_uid": row.accident_uid,
                "year": row.year,
                "edge_id": row.edge_id,
                "edge_length_m": edge_length,
                "snap_distance_m": row.snap_distance_m,
                "distance_to_nearest_intersection_m": dist_to_nearest_intersection,
                "fraction_of_road_length_to_nearest_intersection": frac,
            }
        )

    return pd.DataFrame(rows)


def plot_fraction_histogram(fraction_df, save_path):
    """
    Plot the distribution of accident positions along road segments.
    """
    values = fraction_df["fraction_of_road_length_to_nearest_intersection"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=20, edgecolor="black")
    ax.set_xlabel("Fraction of road length to nearest intersection")
    ax.set_ylabel("Number of accidents")
    ax.set_title("Where accidents occur along road segments")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_fraction_summary(fraction_df):
    """
    Summarise whether accidents are nearer intersections or road middles.
    """
    values = fraction_df["fraction_of_road_length_to_nearest_intersection"].dropna()

    if values.empty:
        return pd.DataFrame(
            [{
                "n_accidents_with_valid_fraction": 0,
                "mean_fraction": np.nan,
                "median_fraction": np.nan,
                "share_within_0_25": np.nan,
                "share_within_0_10": np.nan,
            }]
        )

    summary = {
        "n_accidents_with_valid_fraction": int(values.shape[0]),
        "mean_fraction": float(values.mean()),
        "median_fraction": float(values.median()),
        "share_within_0_25": float((values <= 0.25).mean()),
        "share_within_0_10": float((values <= 0.10).mean()),
    }
    return pd.DataFrame([summary])


def save_gdf_csv(gdf, path):
    """
    Save a GeoDataFrame to CSV with geometry converted to WKT.
    """
    df = gdf.copy()
    if "geometry" in df.columns:
        df["geometry_wkt"] = df.geometry.to_wkt()
        df = df.drop(columns="geometry")
    df.to_csv(path, index=False)


def save_outputs(
    edges_with_counts_gdf,
    snapped_accidents_gdf,
    fraction_df,
    fraction_summary_df,
    k_df,
    moran,
):
    save_gdf_csv(edges_with_counts_gdf, OUTPUT_DIR / "taskB_edges_with_accident_counts.csv")
    save_gdf_csv(snapped_accidents_gdf, OUTPUT_DIR / "taskB_snapped_accidents.csv")
    fraction_df.to_csv(OUTPUT_DIR / "taskB_fraction_to_nearest_intersection.csv", index=False)
    fraction_summary_df.to_csv(OUTPUT_DIR / "taskB_fraction_summary.csv", index=False)

    moran_df = pd.DataFrame(
        [{
            "morans_i": float(moran.I),
            "expected_i": float(moran.EI),
            "z_score": float(moran.z_sim),
            "p_value_permutation": float(moran.p_sim),
        }]
    )
    moran_df.to_csv(OUTPUT_DIR / "taskB_morans_i.csv", index=False)
    k_df.to_csv(OUTPUT_DIR / "taskB_k_function.csv", index=False)


def main():
    print("=" * 78)
    print("Task B - Road accidents on the Leeds central road network")
    print("=" * 78)

    G = download_drive_network()
    G_proj = project_network_to_bng(G)
    nodes_gdf, edges_gdf = get_network_gdfs(G_proj)

    accidents_gdf = load_accidents_in_selected_area()

    plot_accident_distribution(
        edges_gdf,
        accidents_gdf,
        OUTPUT_DIR / "taskB_accident_distribution.png",
    )

    snapped_accidents_gdf = snap_accidents_to_nearest_edges(accidents_gdf, edges_gdf)
    edges_with_counts_gdf = compute_edge_accident_counts(edges_gdf, snapped_accidents_gdf)

    plot_edge_accident_counts(
        edges_with_counts_gdf,
        OUTPUT_DIR / "taskB_edge_accident_counts.png",
    )

    moran = compute_morans_i(edges_with_counts_gdf)

    _, k_result = compute_network_k_function(edges_gdf, accidents_gdf)
    k_df = save_k_function_table(k_result, OUTPUT_DIR / "taskB_k_function.csv")
    plot_k_function(k_df, OUTPUT_DIR / "taskB_k_function.png")

    snapped_with_geometry = snapped_accidents_gdf.merge(
        edges_gdf[["edge_id", "u", "v", "geometry"]].rename(
            columns={
                "u": "edge_u",
                "v": "edge_v",
                "geometry": "edge_geometry",
            }
        ),
        on="edge_id",
        how="left",
    )

    fraction_df = compute_fraction_to_nearest_intersection(snapped_with_geometry, nodes_gdf)
    fraction_summary_df = build_fraction_summary(fraction_df)
    plot_fraction_histogram(
        fraction_df,
        OUTPUT_DIR / "taskB_fraction_histogram.png",
    )

    save_outputs(
        edges_with_counts_gdf,
        snapped_accidents_gdf,
        fraction_df,
        fraction_summary_df,
        k_df,
        moran,
    )

    print("\nMain Task B results:")
    print(f"Selected area: {SELECTED_AREA_NAME}")
    print(f"Accidents analysed: {len(accidents_gdf):,}")
    print(f"Moran's I: {moran.I:.6f}")
    print(f"Moran permutation p-value: {moran.p_sim:.6f}")

    above_upper = int((k_df["observed_k"] > k_df["upper_envelope"]).sum())
    below_lower = int((k_df["observed_k"] < k_df["lower_envelope"]).sum())
    print(f"K-function steps above upper envelope: {above_upper}")
    print(f"K-function steps below lower envelope: {below_lower}")

    print("\nFraction-to-intersection summary:")
    print(fraction_summary_df.to_string(index=False))

    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()