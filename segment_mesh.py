#Author: Bruno Zoller
#Institution: University of Stuttgart

import trimesh
import numpy as np
import networkx as nx
import random
import os


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
    avg_face_size = np.mean([np.linalg.norm(n) for n in face_normals])

    for f1, f2 in mesh.face_adjacency:
        p1, p2 = face_centers[f1], face_centers[f2]
        n1, n2 = face_normals[f1], face_normals[f2]

        spatial_dist = np.linalg.norm(p1 - p2)
        normal_diff = np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0))

        if normal_diff > max_normal_angle:
            continue  # Remove very steep edges

        curvature_penalty = np.exp(curvature_penalty_strength * normal_diff)
        spatial_penalty = 1 + (spatial_dist / avg_face_size) ** 2
        weight = spatial_penalty * curvature_penalty

        G.add_edge(f1, f2, weight=weight)

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
    return seed_faces


def segment_mesh(mesh, G, seed_faces):
    """
    Perform geodesic propagation to segment the mesh.
    """
    face_labels = np.full(len(mesh.faces), -1)
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
    print("Segmentation Started");
    mesh_path = r"input\example.obj"
    output_dir = r"output"
    n_seeds = 10 #Amount of Segments
    curvature_penalty_strength = 25.0 #Angle Punishment 

    mesh = load_and_clean_mesh(mesh_path)
    G = build_adjacency_graph(mesh, curvature_penalty_strength)
    seed_faces = select_seeds(mesh.triangles_center, n_seeds)
    print(f"Using seed face indices: {seed_faces}") #Should be implemented to also be chosen manually
    face_labels = segment_mesh(mesh, G, seed_faces)
    export_segments(mesh, face_labels, n_seeds, output_dir)
    print("Segmentation complete.")


if __name__ == "__main__":
    main()
