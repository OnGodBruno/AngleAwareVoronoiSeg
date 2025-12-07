"""
Automatic seed placement using efficient facility location algorithms.

This module implements various facility placement algorithms for optimizing
seed placement in mesh segmentation. The goal is to minimize the maximum
geodesic distance (with penalties) from any face to its nearest seed.
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.csgraph import dijkstra
from typing import List, Tuple, Optional, Union
import time
import warnings
from dataclasses import dataclass
from enum import Enum


class PlacementStrategy(Enum):
    """Available facility placement strategies."""
    GREEDY_MINIMAX = "greedy_minimax"
    KMEANS_PLUS_PLUS = "kmeans_plus_plus"
    FARTHEST_FIRST = "farthest_first"
    GONZALEZ_APPROXIMATION = "gonzalez_approximation"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


@dataclass
class PlacementConfig:
    """Configuration for facility placement algorithms."""
    strategy: PlacementStrategy = PlacementStrategy.ADAPTIVE_HYBRID
    max_computation_time: float = 30.0  # Maximum computation time in seconds
    distance_threshold: float = 1e6  # Threshold for considering points unreachable
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100
    random_seed: int = 42
    verbose: bool = True


@dataclass
class PlacementResult:
    """Result of facility placement algorithm."""
    seed_indices: np.ndarray
    max_distance: float
    computation_time: float
    strategy_used: PlacementStrategy
    convergence_info: dict


class FacilityPlacer:
    """
    Main class for automatic seed placement using facility location algorithms.
    
    The goal is to solve the p-center problem: given a graph with weighted edges
    (representing geodesic distances with penalties), place k facilities (seeds)
    to minimize the maximum distance from any vertex to its nearest facility.
    """

    def __init__(self, config: PlacementConfig = None):
        """Initialize the facility placer with configuration."""
        self.config = config or PlacementConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def automatic_seed_placement(
        self,
        adjacency_graph: sparse.csr_matrix,
        num_seeds: int,
        face_centers: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Main entry point for automatic seed placement.
        
        Args:
            adjacency_graph: Sparse adjacency matrix with penalties applied
            num_seeds: Number of seeds to place
            face_centers: Optional face center coordinates for spatial algorithms
            
        Returns:
            Array of seed indices that minimize the maximum penalty distance
        """
        if num_seeds <= 0:
            raise ValueError("Number of seeds must be positive")

        if num_seeds >= adjacency_graph.shape[0]:
            # If we need more seeds than faces, just return all face indices
            return np.arange(adjacency_graph.shape[0])

        start_time = time.time()

        # Choose strategy based on problem size and configuration
        strategy = self._choose_strategy(adjacency_graph, num_seeds)

        if self.config.verbose:
            print(f"Using strategy: {strategy.value} for {num_seeds} seeds on {adjacency_graph.shape[0]} faces")

        # Execute the chosen strategy
        result = self._execute_strategy(strategy, adjacency_graph, num_seeds, face_centers)
        computation_time = time.time() - start_time

        if self.config.verbose:
            print(f"Placement completed in {computation_time:.2f}s, max distance: {result.max_distance:.3f}")

        return result.seed_indices

    def _choose_strategy(self, adjacency_graph: sparse.csr_matrix, num_seeds: int) -> PlacementStrategy:
        """Choose the best strategy based on problem characteristics."""
        n_faces = adjacency_graph.shape[0]
        n_edges = adjacency_graph.nnz

        # Estimate computational complexity
        greedy_complexity = num_seeds * n_faces * n_faces  # O(k * n^2)
        gonzalez_complexity = num_seeds * n_faces  # O(k * n)

        if self.config.strategy != PlacementStrategy.ADAPTIVE_HYBRID:
            return self.config.strategy

        # Adaptive strategy selection
        if n_faces < 1000:
            # Small mesh: use exact greedy algorithm
            return PlacementStrategy.GREEDY_MINIMAX
        elif n_faces < 10000 and num_seeds <= 20:
            # Medium mesh with few seeds: use Gonzalez approximation
            return PlacementStrategy.GONZALEZ_APPROXIMATION
        else:
            # Large mesh: use faster heuristics
            return PlacementStrategy.FARTHEST_FIRST

    def _execute_strategy(
        self,
        strategy: PlacementStrategy,
        adjacency_graph: sparse.csr_matrix,
        num_seeds: int,
        face_centers: Optional[np.ndarray]
    ) -> PlacementResult:
        """Execute the specified placement strategy."""

        strategies = {
            PlacementStrategy.GREEDY_MINIMAX: self._greedy_minimax_placement,
            PlacementStrategy.GONZALEZ_APPROXIMATION: self._gonzalez_approximation,
            PlacementStrategy.FARTHEST_FIRST: self._farthest_first_placement,
            PlacementStrategy.KMEANS_PLUS_PLUS: self._kmeans_plus_plus_placement,
        }

        if strategy not in strategies:
            strategy = PlacementStrategy.GONZALEZ_APPROXIMATION

        return strategies[strategy](adjacency_graph, num_seeds, face_centers)

    def _greedy_minimax_placement(
        self,
        adjacency_graph: sparse.csr_matrix,
        num_seeds: int,
        face_centers: Optional[np.ndarray]
    ) -> PlacementResult:
        """
        Greedy algorithm for the p-center problem.

        At each step, add the facility that minimizes the maximum distance
        to all points. This gives a 2-approximation for the p-center problem.
        """
        start_time = time.time()
        n_faces = adjacency_graph.shape[0]

        # Start with a random seed or the most central one
        if face_centers is not None:
            # Choose the most central face (closest to centroid)
            centroid = np.mean(face_centers, axis=0)
            distances_to_centroid = np.linalg.norm(face_centers - centroid, axis=1)
            current_seeds = [np.argmin(distances_to_centroid)]
        else:
            current_seeds = [self.rng.choice(n_faces)]

        convergence_info = {"iterations": [], "max_distances": []}

        for iteration in range(1, num_seeds):
            if time.time() - start_time > self.config.max_computation_time:
                warnings.warn(f"Computation time limit exceeded, stopping at {iteration} seeds")
                break

            best_candidate = None
            best_max_distance = float('inf')

            # Calculate current distances from all points to nearest facility
            current_distances = self._compute_distances_to_facilities(
                adjacency_graph, current_seeds
            )

            # Try each remaining face as a potential new facility
            candidates = list(set(range(n_faces)) - set(current_seeds))

            # For efficiency, sample candidates if there are too many
            if len(candidates) > 1000:
                candidates = self.rng.choice(candidates, size=1000, replace=False).tolist()

            for candidate in candidates:
                # Compute distances to this candidate
                candidate_distances = self._compute_single_source_distances(
                    adjacency_graph, candidate
                )

                # Compute new maximum distance with this candidate added
                new_distances = np.minimum(current_distances, candidate_distances)
                max_distance = np.max(new_distances[np.isfinite(new_distances)])

                if max_distance < best_max_distance:
                    best_max_distance = max_distance
                    best_candidate = candidate

            if best_candidate is not None:
                current_seeds.append(best_candidate)
                convergence_info["iterations"].append(iteration)
                convergence_info["max_distances"].append(best_max_distance)

                if self.config.verbose:
                    print(f"  Iteration {iteration}: added seed {best_candidate}, max distance: {best_max_distance:.3f}")

        final_distances = self._compute_distances_to_facilities(adjacency_graph, current_seeds)
        max_distance = np.max(final_distances[np.isfinite(final_distances)])

        return PlacementResult(
            seed_indices=np.array(current_seeds, dtype=int),
            max_distance=max_distance,
            computation_time=time.time() - start_time,
            strategy_used=PlacementStrategy.GREEDY_MINIMAX,
            convergence_info=convergence_info
        )

    def _gonzalez_approximation(
        self,
        adjacency_graph: sparse.csr_matrix,
        num_seeds: int,
        face_centers: Optional[np.ndarray]
    ) -> PlacementResult:
        """
        Gonzalez's 2-approximation algorithm for the p-center problem.
        
        This is much faster than the greedy algorithm and still provides
        a 2-approximation guarantee.
        """
        start_time = time.time()
        n_faces = adjacency_graph.shape[0]

        # Start with a random seed
        current_seeds = [self.rng.choice(n_faces)]
        convergence_info = {"iterations": [], "max_distances": []}

        for iteration in range(1, num_seeds):
            # Find the point that is farthest from all current facilities
            current_distances = self._compute_distances_to_facilities(
                adjacency_graph, current_seeds
            )

            # Find the point with maximum distance to nearest facility
            valid_distances = current_distances[np.isfinite(current_distances)]
            if len(valid_distances) == 0:
                break

            farthest_point = np.argmax(current_distances)
            current_seeds.append(farthest_point)

            max_distance = current_distances[farthest_point]
            convergence_info["iterations"].append(iteration)
            convergence_info["max_distances"].append(max_distance)

            if self.config.verbose:
                print(f"  Gonzalez iteration {iteration}: added seed {farthest_point}, distance: {max_distance:.3f}")

        # Compute final maximum distance
        final_distances = self._compute_distances_to_facilities(adjacency_graph, current_seeds)
        max_distance = np.max(final_distances[np.isfinite(final_distances)])

        return PlacementResult(
            seed_indices=np.array(current_seeds, dtype=int),
            max_distance=max_distance,
            computation_time=time.time() - start_time,
            strategy_used=PlacementStrategy.GONZALEZ_APPROXIMATION,
            convergence_info=convergence_info
        )

    def _farthest_first_placement(
        self,
        adjacency_graph: sparse.csr_matrix,
        num_seeds: int,
        face_centers: Optional[np.ndarray]
    ) -> PlacementResult:
        """
        Simple farthest-first placement algorithm.

        Similar to Gonzalez but uses spatial distances when available
        for efficiency on large meshes.
        """
        start_time = time.time()
        n_faces = adjacency_graph.shape[0]

        # If we have face centers, use spatial distances for efficiency
        if face_centers is not None:
            return self._spatial_farthest_first(face_centers, num_seeds, start_time)

        # Otherwise, use graph distances (slower but more accurate)
        return self._graph_farthest_first(adjacency_graph, num_seeds, start_time)

    def _spatial_farthest_first(
        self,
        face_centers: np.ndarray,
        num_seeds: int,
        start_time: float
    ) -> PlacementResult:
        """Farthest-first using spatial distances."""
        n_faces = face_centers.shape[0]

        # Start with a random seed
        current_seeds = [self.rng.choice(n_faces)]
        convergence_info = {"iterations": [], "max_distances": []}

        # Initialize distances
        distances = np.full(n_faces, np.inf)

        for iteration in range(1, num_seeds):
            # Update distances to current facilities
            for seed in current_seeds:
                seed_distances = np.linalg.norm(face_centers - face_centers[seed], axis=1)
                distances = np.minimum(distances, seed_distances)

            # Find farthest point
            farthest_point = np.argmax(distances)
            current_seeds.append(farthest_point)

            max_distance = distances[farthest_point]
            convergence_info["iterations"].append(iteration)
            convergence_info["max_distances"].append(max_distance)

        # Final distance calculation
        distances = np.full(n_faces, np.inf)
        for seed in current_seeds:
            seed_distances = np.linalg.norm(face_centers - face_centers[seed], axis=1)
            distances = np.minimum(distances, seed_distances)

        max_distance = np.max(distances)

        return PlacementResult(
            seed_indices=np.array(current_seeds, dtype=int),
            max_distance=max_distance,
            computation_time=time.time() - start_time,
            strategy_used=PlacementStrategy.FARTHEST_FIRST,
            convergence_info=convergence_info
        )

    def _graph_farthest_first(
        self,
        adjacency_graph: sparse.csr_matrix,
        num_seeds: int,
        start_time: float
    ) -> PlacementResult:
        """Farthest-first using graph distances."""
        n_faces = adjacency_graph.shape[0]

        # Start with a random seed
        current_seeds = [self.rng.choice(n_faces)]
        convergence_info = {"iterations": [], "max_distances": []}

        for iteration in range(1, num_seeds):
            if time.time() - start_time > self.config.max_computation_time:
                warnings.warn(f"Time limit exceeded at iteration {iteration}")
                break

            # Find current distances to all facilities
            distances = self._compute_distances_to_facilities(adjacency_graph, current_seeds)

            # Find farthest point
            valid_mask = np.isfinite(distances)
            if not np.any(valid_mask):
                break

            farthest_point = np.argmax(distances[valid_mask])
            farthest_point = np.where(valid_mask)[0][farthest_point]

            current_seeds.append(farthest_point)

            max_distance = distances[farthest_point]
            convergence_info["iterations"].append(iteration)
            convergence_info["max_distances"].append(max_distance)

        # Final distance calculation
        final_distances = self._compute_distances_to_facilities(adjacency_graph, current_seeds)
        max_distance = np.max(final_distances[np.isfinite(final_distances)])

        return PlacementResult(
            seed_indices=np.array(current_seeds, dtype=int),
            max_distance=max_distance,
            computation_time=time.time() - start_time,
            strategy_used=PlacementStrategy.FARTHEST_FIRST,
            convergence_info=convergence_info
        )

    def _kmeans_plus_plus_placement(
        self,
        adjacency_graph: sparse.csr_matrix,
        num_seeds: int,
        face_centers: Optional[np.ndarray]
    ) -> PlacementResult:
        """
        K-means++ style seed placement using graph distances.
        
        Seeds are placed with probability proportional to their distance
        from existing facilities.
        """
        start_time = time.time()
        n_faces = adjacency_graph.shape[0]

        # Start with a random seed
        current_seeds = [self.rng.choice(n_faces)]
        convergence_info = {"iterations": [], "max_distances": []}

        for iteration in range(1, num_seeds):
            if time.time() - start_time > self.config.max_computation_time:
                warnings.warn(f"Time limit exceeded at iteration {iteration}")
                break

            # Compute distances to current facilities
            distances = self._compute_distances_to_facilities(adjacency_graph, current_seeds)

            # Use distances as probabilities (squared for k-means++ style)
            valid_mask = np.isfinite(distances)
            if not np.any(valid_mask):
                break

            probabilities = distances[valid_mask] ** 2
            probabilities /= np.sum(probabilities)

            # Sample next seed
            valid_indices = np.where(valid_mask)[0]
            next_seed_idx = self.rng.choice(len(valid_indices), p=probabilities)
            next_seed = valid_indices[next_seed_idx]

            current_seeds.append(next_seed)

            max_distance = np.max(distances[valid_mask])
            convergence_info["iterations"].append(iteration)
            convergence_info["max_distances"].append(max_distance)

        # Final distance calculation
        final_distances = self._compute_distances_to_facilities(adjacency_graph, current_seeds)
        max_distance = np.max(final_distances[np.isfinite(final_distances)])

        return PlacementResult(
            seed_indices=np.array(current_seeds, dtype=int),
            max_distance=max_distance,
            computation_time=time.time() - start_time,
            strategy_used=PlacementStrategy.KMEANS_PLUS_PLUS,
            convergence_info=convergence_info
        )

    def _compute_distances_to_facilities(
        self,
        adjacency_graph: sparse.csr_matrix,
        facility_indices: List[int]
    ) -> np.ndarray:
        """
        Compute the distance from each point to its nearest facility.
        
        Args:
            adjacency_graph: Weighted adjacency matrix
            facility_indices: List of facility (seed) indices
            
        Returns:
            Array of distances from each point to nearest facility
        """
        if not facility_indices:
            return np.full(adjacency_graph.shape[0], np.inf)

        # Use multi-source Dijkstra for efficiency
        try:
            distances_matrix = dijkstra(
                adjacency_graph,
                indices=facility_indices,
                directed=False,
                return_predecessors=False
            )

            # Return minimum distance to any facility
            return np.min(distances_matrix, axis=0)

        except Exception as e:
            warnings.warn(f"Dijkstra computation failed: {e}, using fallback")
            return self._compute_distances_fallback(adjacency_graph, facility_indices)

    def _compute_single_source_distances(
        self,
        adjacency_graph: sparse.csr_matrix,
        source: int
    ) -> np.ndarray:
        """Compute distances from a single source to all other points."""
        try:
            return dijkstra(
                adjacency_graph,
                indices=[source],
                directed=False,
                return_predecessors=False
            )[0]
        except Exception as e:
            warnings.warn(f"Single-source Dijkstra failed: {e}")
            return np.full(adjacency_graph.shape[0], np.inf)

    def _compute_distances_fallback(
        self,
        adjacency_graph: sparse.csr_matrix,
        facility_indices: List[int]
    ) -> np.ndarray:
        """Fallback distance computation for when Dijkstra fails."""
        n_points = adjacency_graph.shape[0]
        min_distances = np.full(n_points, np.inf)

        for facility in facility_indices:
            try:
                distances = dijkstra(
                    adjacency_graph,
                    indices=[facility],
                    directed=False,
                    return_predecessors=False
                )[0]
                min_distances = np.minimum(min_distances, distances)
            except:
                continue

        return min_distances


# Convenience function for direct usage
def automatic_seed_placement(
    adjacency_graph: sparse.csr_matrix,
    num_seeds: int,
    face_centers: Optional[np.ndarray] = None,
    strategy: Union[str, PlacementStrategy] = "adaptive_hybrid",
    max_computation_time: float = 30.0,
    verbose: bool = True
) -> np.ndarray:
    """
    Automatically place seeds to minimize maximum penalty distance.
    
    This function implements efficient facility placement algorithms to solve
    the p-center problem on mesh segmentation graphs.
    
    Args:
        adjacency_graph: Sparse adjacency matrix with penalties applied
                        Shape: (n_faces, n_faces)
        num_seeds: Number of seeds to place
        face_centers: Optional face center coordinates for spatial algorithms
                     Shape: (n_faces, 3)
        strategy: Placement strategy to use. Options:
                 - "adaptive_hybrid": Automatically choose best strategy
                 - "greedy_minimax": Exact greedy algorithm (slow but optimal)
                 - "gonzalez_approximation": Fast 2-approximation
                 - "farthest_first": Simple farthest-first heuristic
                 - "kmeans_plus_plus": Probabilistic placement
        max_computation_time: Maximum time to spend computing (seconds)
        verbose: Whether to print progress information
        
    Returns:
        Array of seed face indices that minimize maximum penalty distance
        
    Example:
        >>> # Assuming you have an adjacency graph with penalties
        >>> seed_indices = automatic_seed_placement(
        ...     adjacency_graph=sparse_matrix,
        ...     num_seeds=10,
        ...     face_centers=face_centers,
        ...     strategy="adaptive_hybrid"
        ... )
    """

    # Convert string strategy to enum
    if isinstance(strategy, str):
        strategy_map = {
            "adaptive_hybrid": PlacementStrategy.ADAPTIVE_HYBRID,
            "greedy_minimax": PlacementStrategy.GREEDY_MINIMAX,
            "gonzalez_approximation": PlacementStrategy.GONZALEZ_APPROXIMATION,
            "farthest_first": PlacementStrategy.FARTHEST_FIRST,
            "kmeans_plus_plus": PlacementStrategy.KMEANS_PLUS_PLUS,
        }
        strategy = strategy_map.get(strategy, PlacementStrategy.ADAPTIVE_HYBRID)

    # Create configuration
    config = PlacementConfig(
        strategy=strategy,
        max_computation_time=max_computation_time,
        verbose=verbose
    )

    # Create placer and run algorithm
    placer = FacilityPlacer(config)
    return placer.automatic_seed_placement(adjacency_graph, num_seeds, face_centers)