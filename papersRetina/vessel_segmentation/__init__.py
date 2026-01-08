"""
Retinal Vessel Segmentation Library

Fast retinal vessel detection and measurement using wavelets and edge location refinement.

Based on: Bankhead P, Scholfield CN, McGeown JG, Curtis TM (2012)
"Fast Retinal Vessel Detection and Measurement Using Wavelets and Edge Location Refinement"
PLoS ONE 7(3): e32435

Usage:
    from vessel_segmentation import analyze_retinal_image, RetinalVesselAnalyzer
    
    result = analyze_retinal_image(image)
"""

from .core import (
    IUWT,
    VesselSegmenter,
    CentrelineExtractor,
    ProfileGenerator,
    EdgeDetector,
    RetinalVesselAnalyzer,
    VesselSegment,
    SegmentationResult,
    analyze_retinal_image,
)

from .visualization import (
    show_segmentation,
    show_wavelet_levels,
    show_centrelines,
    show_vessel_profile,
    show_diameter_histogram,
    show_processing_steps,
    create_edge_overlay,
)

__version__ = "1.0.0"
__author__ = "Based on Bankhead et al. (2012)"

__all__ = [
    # Core classes
    "IUWT",
    "VesselSegmenter",
    "CentrelineExtractor",
    "ProfileGenerator",
    "EdgeDetector",
    "RetinalVesselAnalyzer",
    # Data classes
    "VesselSegment",
    "SegmentationResult",
    # Convenience functions
    "analyze_retinal_image",
    # Visualization
    "show_segmentation",
    "show_wavelet_levels",
    "show_centrelines",
    "show_vessel_profile",
    "show_diameter_histogram",
    "show_processing_steps",
    "create_edge_overlay",
]
