"""
Automatic seed placement using efficient facility location algorithms.

This module is the renamed version of facility_placement.py and provides
the same functionality with improved facility placement algorithms.
"""

# Import all functionality from facility_placement for backwards compatibility
from .facility_placement import (
    FacilityPlacer,
    PlacementStrategy,
    PlacementConfig,
    PlacementResult,
    automatic_seed_placement
)

__all__ = [
    'FacilityPlacer',
    'PlacementStrategy',
    'PlacementConfig',
    'PlacementResult',
    'automatic_seed_placement'
]