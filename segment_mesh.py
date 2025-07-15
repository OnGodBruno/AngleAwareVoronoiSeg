#Author: Bruno Zoller
#Institution: University of Stuttgart
import networkx
import trimesh
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
import random
import os
import argparse


def load_and_clean_mesh(mesh_path):
    """
    Load and clean a 3D mesh.
    """
    mesh = trimesh.load(mesh_path, process=True)
    mesh.remove_unreferenced_vertices()
    mesh.remove_infinite_values()
    return mesh


def build_adjacency_graph(mesh, curvature_penalty_strength, max_normal_angle=np.radians(20)):
    """
    Build a face adjacency graph with curvature-aware edge weights.
    """
    G = nx.Graph()
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    edge_lengths = mesh.edges_unique_length

    avg_edge_length = edge_lengths.mean()

    for f1, f2 in mesh.face_adjacency:
        p1, p2 = face_centers[f1], face_centers[f2]
        n1, n2 = face_normals[f1], face_normals[f2]

        spatial_dist = np.linalg.norm(p1 - p2)
        normal_diff = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))

        if normal_diff > max_normal_angle:
            continue  # Remove very steep edges

        curvature_penalty = np.exp(curvature_penalty_strength * normal_diff)
        spatial_penalty = 1 + (spatial_dist / avg_edge_length) ** 2
        weight = spatial_penalty * curvature_penalty

        G.add_edge(f1, f2, weight=weight)
    print("build_adjacency_graph finished")  # DEBUG
    return G


def select_seeds(face_centers, n_seeds, graph_nodes):
    """
    Select seed faces using farthest-point sampling.
    """
    rng = np.random.default_rng()
    n_faces = face_centers.shape[0]

    seed_faces_id = [rng.integers(n_faces)]
    d_min = np.linalg.norm(face_centers - face_centers[seed_faces_id[0]], axis=1)

    for _ in range(1, n_seeds):
        probs = d_min / d_min.sum()
        new_seed = rng.choice(n_faces, p=probs)
        seed_faces_id.append(new_seed)

        d_new = np.linalg.norm(face_centers - face_centers[new_seed], axis=1)
        d_min = np.minimum(d_min, d_new)

    seed_faces = graph_nodes[seed_faces_id]
    print("select_seeds finished")  # DEBUG
    return seed_faces


def segment_mesh(G, seed_faces):
    """
    Perform geodesic propagation to segment the mesh.
    """
    nodelist = np.asarray(list(G), dtype=int)
    node2idx = {v: i for i, v in enumerate(nodelist)}
    seed_idx = np.fromiter((node2idx[s] for s in seed_faces), int)

    # CSR-Sparse-Matrix, necessary for csgraph.dijkstra()
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist,
                                 weight="weight", dtype=np.float64)

    # Multi-source dijkstra on all seed nodes in G (seed_idx)
    dist = csgraph.dijkstra(A, indices=seed_idx,
                            directed=False, return_predecessors=False)

    # Generate offsets for each seed so that the earliest seed wins on exact distance matches
    eps = np.linspace(0.0, 1e-9, len(seed_idx), endpoint=False)[:, None]
    winner = (dist + eps).argmin(axis=0)

    face_labels = {nodelist[i]: seed_faces[winner[i]]
                   for i in range(len(nodelist))
                   if np.isfinite(dist[winner[i], i])}

    print("segment_mesh finished")  # DEBUG
    return face_labels


def export_segments(mesh, face_labels, seed_faces, output_dir):
    """
    Export segmented mesh parts as separate OBJ files.
    """
    os.makedirs(output_dir, exist_ok=True)
    seed_to_idx = {seed_id: i for i, seed_id in enumerate(seed_faces)}
    n_seeds = len(seed_faces)

    segments = [[] for _ in range(n_seeds)]
    for f_idx, seed_id in face_labels.items():
        segment_idx = seed_to_idx[seed_id]
        segments[segment_idx].append(f_idx)

    for i, face_ids in enumerate(segments):
        if face_ids:
            part = mesh.submesh([face_ids], append=True)
            part.export(os.path.join(output_dir, f"segment_{i}.obj"))


def main():
    parser = argparse.ArgumentParser(description="3D Mesh Segmentation")
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=r"input\run\example.obj",
        help="Path to mesh file (default: input\\run\\example.obj)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to output directory (default: output)",
    )
    parser.add_argument("--n_seeds", type=int, default=10, help="Number of segments")
    parser.add_argument(
        "--curvature_penalty_strength",
        type=float,
        default=100.0,
        help="Angle punishment strength",
    )
    parser.add_argument(
        "--seed_faces",
        type=int,
        nargs="*",
        help="Manual seed face indices (optional)",
    )

    args = parser.parse_args()
    print("Segmentation Started")

    mesh = load_and_clean_mesh(args.mesh_path)
    G = build_adjacency_graph(mesh, args.curvature_penalty_strength)
    graph_nodes = np.array(list(G.nodes()), dtype=int)
    face_centers_in_G = mesh.triangles_center[graph_nodes]

    if args.seed_faces is None:
        seed_faces = select_seeds(face_centers_in_G, args.n_seeds, graph_nodes)
    else:
        seed_faces = np.array(args.seed_faces, dtype=int)

    print(f"Using seed face indices: {seed_faces}")
    face_labels = segment_mesh(G, seed_faces)

    export_segments(mesh, face_labels, seed_faces, args.output_dir)
    print("Segmentation complete.")


if __name__ == "__main__":
    main()
