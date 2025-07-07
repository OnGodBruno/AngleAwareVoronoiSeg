#Author: Bruno Zoller
#Institution: University of Stuttgart

import trimesh
import numpy as np
import networkx as nx
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


def select_seeds(face_centers, n_seeds):
    """
    Select seed faces using farthest-point sampling.
    """
    seed_faces = [random.randint(0, len(face_centers) - 1)]
    while len(seed_faces) < n_seeds:
        dists = [
            min(np.linalg.norm(face_centers[i] - face_centers[s]) for s in seed_faces)
            for i in range(len(face_centers))
        ]
        probs = np.array(dists) / np.sum(dists)
        seed_faces.append(np.random.choice(len(face_centers), p=probs))

    print("select_seeds finished")
    return seed_faces


def segment_mesh(mesh, G, seed_faces):
    """
    Perform geodesic propagation to segment the mesh.
    """
    face_labels = np.full(len(mesh.faces), -1)

    # stores the current seed and distance for each face: {f_idx: (seed, dist)}
    distance_map = {}

    for seed_id, seed in enumerate(seed_faces):
        lengths = nx.single_source_dijkstra_path_length(G, seed)
        for f_idx, dist in lengths.items():
            if face_labels[f_idx] == -1 or dist < distance_map[f_idx][1]:
                face_labels[f_idx] = seed_id
                distance_map[f_idx] = (seed, dist)

    return face_labels


def export_segments(mesh, face_labels, n_seeds, output_dir):
    """
    Export segmented mesh parts as separate OBJ files.
    """
    os.makedirs(output_dir, exist_ok=True)
    segments = [[] for _ in range(n_seeds)]
    for f_idx, label in enumerate(face_labels):
        if label >= 0:
            segments[label].append(f_idx)

    for i, face_ids in enumerate(segments):
        if face_ids:
            part = mesh.submesh([face_ids], append=True)
            part.export(os.path.join(output_dir, f"segment_{i}.obj"))


def main():
    parser = argparse.ArgumentParser(description="3D Mesh Segmentation")
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=r"input\run\mesh.obj",
        help="Path to mesh file (default: input\\run\\mesh.obj)",
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
        default=25.0,
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

    seed_faces = args.seed_faces or select_seeds(mesh.triangles_center, args.n_seeds)
    n_seeds = len(seed_faces)

    print(f"Using seed face indices: {seed_faces}")
    face_labels = segment_mesh(mesh, G, seed_faces)
    export_segments(mesh, face_labels, n_seeds, args.output_dir)
    print("Segmentation complete.")


if __name__ == "__main__":
    main()
