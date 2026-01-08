# Retinal Vessel Segmentation Library

A Python implementation of retinal vessel detection and measurement using wavelets and edge location refinement.

Based on the algorithm described in:
> Bankhead P, Scholfield CN, McGeown JG, Curtis TM (2012) **"Fast Retinal Vessel Detection and Measurement Using Wavelets and Edge Location Refinement"**. PLoS ONE 7(3): e32435. doi:10.1371/journal.pone.0032435

## Features

- **IUWT-based vessel segmentation**: Fast, unsupervised vessel detection using the Isotropic Undecimated Wavelet Transform
- **Centreline extraction**: Morphological thinning with spline fitting for smooth vessel centrelines
- **Edge detection**: Sub-pixel accurate edge detection using zero-crossings of the second derivative
- **Diameter measurement**: Automatic vessel diameter measurements along the entire vessel length
- **Visualization tools**: Comprehensive visualization utilities for results analysis

## Algorithm Overview

The algorithm consists of several main steps:

1. **Vessel Segmentation (IUWT)**
   - Apply the Isotropic Undecimated Wavelet Transform
   - Sum selected wavelet levels (typically 2-3 for fundus images)
   - Threshold based on percentage of lowest/highest coefficients
   - Clean binary image (remove small objects, fill holes)

2. **Centreline Extraction**
   - Morphological thinning to obtain skeletonized vessels
   - Remove branch points to separate vessel segments
   - Remove spurs and short segments
   - Fit cubic splines for smooth centrelines

3. **Image Profile Generation**
   - Generate intensity profiles perpendicular to vessel at each centreline point
   - Create "straightened" vessel images

4. **Edge Detection**
   - Estimate vessel width from initial segmentation
   - Apply anisotropic Gaussian smoothing
   - Compute second derivative perpendicular to vessel
   - Find zero-crossings and connect into continuous edge trails
   - Calculate diameters as distance between edges

## Installation

```bash
# Clone the repository
git clone https://github.com/example/retinal-vessel-segmentation.git
cd retinal-vessel-segmentation

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

```python
import cv2
from vessel_segmentation import analyze_retinal_image, RetinalVesselAnalyzer

# Load image
image = cv2.imread('fundus_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Quick analysis
result = analyze_retinal_image(image, dark_vessels=True, wavelet_levels=[2, 3])

# Access results
print(f"Number of vessel segments: {len(result.vessels)}")
print(f"Binary mask shape: {result.binary_mask.shape}")

# Get diameter measurements
analyzer = RetinalVesselAnalyzer()
stats = analyzer.get_statistics(result)
print(f"Mean diameter: {stats['mean_diameter']:.2f} pixels")
```

## Detailed Usage

### Basic Segmentation

```python
from vessel_segmentation import VesselSegmenter

# Initialize segmenter
segmenter = VesselSegmenter(
    dark_vessels=True,      # True for fundus images, False for angiograms
    wavelet_levels=[2, 3],  # Wavelet levels to use
    threshold_percent=0.20, # Percentage of pixels to classify as vessels
    min_object_size_percent=0.05,  # Remove objects smaller than this
    fill_hole_size_percent=0.05    # Fill holes smaller than this
)

# Segment
result = segmenter.segment(image)
binary_mask = result.binary_mask
```

### Full Pipeline Analysis

```python
from vessel_segmentation import RetinalVesselAnalyzer

analyzer = RetinalVesselAnalyzer(
    # Segmentation
    dark_vessels=True,
    wavelet_levels=[2, 3],
    threshold_percent=0.20,
    min_object_size_percent=0.05,
    fill_hole_size_percent=0.05,
    # Centreline extraction
    spur_length=10,
    min_segment_length=10,
    spline_spacing=10,
    # Edge detection
    smooth_parallel=1.0,
    smooth_perpendicular=0.1,
    enforce_connectivity=True
)

result = analyzer.analyze(image)

# Access vessel segments
for i, vessel in enumerate(result.vessels):
    print(f"Vessel {i}: {len(vessel.centre)} points")
    if vessel.diameters is not None:
        valid_d = vessel.diameters[~np.isnan(vessel.diameters)]
        print(f"  Mean diameter: {np.mean(valid_d):.2f} px")
```

### Visualization

```python
from visualization import (
    show_segmentation,
    show_wavelet_levels,
    show_centrelines,
    show_diameter_histogram,
    create_edge_overlay
)

# Show segmentation results
show_segmentation(image, result)

# Show wavelet decomposition
from vessel_segmentation import IUWT
iuwt = IUWT()
wavelet_coeffs, scaling_coeffs = iuwt.transform(gray_image, n_levels=4)
show_wavelet_levels(wavelet_coeffs, scaling_coeffs)

# Show centrelines colored by diameter
show_centrelines(image, result.vessels, color_by_diameter=True)

# Diameter histogram
show_diameter_histogram(result)

# Create overlay image
overlay = create_edge_overlay(image, result)
```

### Running the Demo

```bash
# Run with synthetic test image
python demo.py

# Run with your own image
python demo.py --image path/to/retinal_image.jpg

# Adjust parameters
python demo.py --image fundus.png --levels 3 4 --threshold 0.15

# Save without displaying
python demo.py --image fundus.png --no-show
```

## Parameter Guidelines

### Wavelet Levels

The choice of wavelet levels depends on image resolution:

| Image Type | Typical Size | Recommended Levels |
|------------|--------------|-------------------|
| DRIVE database | 565×584 | [2, 3] |
| Standard fundus | ~1500×1500 | [2, 3] or [3, 4] |
| High resolution | >3000×3000 | [3, 4] or [3, 4, 5] |

Higher wavelet levels capture larger structures. For high-resolution images, use higher levels.

### Threshold

The threshold percentage controls how many pixels are classified as vessels:

- **0.15-0.20**: Good for most fundus images (tends to over-segment slightly)
- **0.10-0.15**: More conservative, may miss fine vessels
- **0.20-0.25**: More aggressive, may include more noise

The algorithm is designed to over-segment initially (around 20%) because true vessel pixels typically make up 12-14% of the FOV. Subsequent cleanup steps remove false positives.

### Dark Vessels

- `dark_vessels=True`: For standard fundus photographs (vessels appear darker than background)
- `dark_vessels=False`: For fluorescein angiograms (vessels appear brighter)

## API Reference

### Classes

#### `IUWT`
Implements the Isotropic Undecimated Wavelet Transform.

#### `VesselSegmenter`
Performs vessel segmentation using IUWT and thresholding.

#### `CentrelineExtractor`
Extracts and refines vessel centrelines.

#### `ProfileGenerator`
Generates image intensity profiles perpendicular to vessels.

#### `EdgeDetector`
Detects vessel edges using zero-crossings.

#### `RetinalVesselAnalyzer`
Complete analysis pipeline combining all components.

### Data Classes

#### `VesselSegment`
```python
@dataclass
class VesselSegment:
    centre: np.ndarray      # Centreline coordinates (N, 2)
    angles: np.ndarray      # Orientation at each point
    side1: np.ndarray       # Left edge coordinates
    side2: np.ndarray       # Right edge coordinates
    diameters: np.ndarray   # Diameter at each point
    im_profiles: np.ndarray # Intensity profiles
```

#### `SegmentationResult`
```python
@dataclass
class SegmentationResult:
    binary_mask: np.ndarray        # Binary segmentation
    wavelet_sum: np.ndarray        # Sum of wavelet coefficients
    vessels: List[VesselSegment]   # Extracted vessels
    fov_mask: np.ndarray           # Field of view mask
```

## Performance

The algorithm is designed for efficiency:

| Image Size | Typical Processing Time |
|------------|------------------------|
| 565×584 (DRIVE) | ~1 second |
| 1360×1024 | ~3-5 seconds |
| 2160×1440 | ~5-7 seconds |
| 3584×2438 | ~20-30 seconds |

Processing time depends on the number of vessels detected and image complexity.

## References

1. Bankhead P, Scholfield CN, McGeown JG, Curtis TM (2012) Fast Retinal Vessel Detection and Measurement Using Wavelets and Edge Location Refinement. PLoS ONE 7(3): e32435.

2. Starck JL, Fadili J, Murtagh F (2007) The undecimated wavelet decomposition and its reconstruction. IEEE Trans Signal Process 16: 297-309.

3. DRIVE Database: http://www.isi.uu.nl/Research/Databases/DRIVE/

4. REVIEW Database: http://reviewdb.lincoln.ac.uk/

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
