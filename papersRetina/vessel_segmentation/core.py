"""
Retinal Vessel Detection and Measurement Library

Implementation based on:
Bankhead P, Scholfield CN, McGeown JG, Curtis TM (2012) 
"Fast Retinal Vessel Detection and Measurement Using Wavelets and Edge Location Refinement"
PLoS ONE 7(3): e32435. doi:10.1371/journal.pone.0032435

This library provides tools for:
1. Vessel segmentation using the Isotropic Undecimated Wavelet Transform (IUWT)
2. Centreline extraction and spline fitting
3. Image profile generation perpendicular to vessels
4. Vessel edge detection using zero-crossings of the second derivative
5. Diameter measurement

Author: Implementation based on Bankhead et al. (2012)
"""

import numpy as np
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, label, binary_fill_holes
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import regionprops, label as sk_label
import cv2
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass, field


@dataclass
class VesselSegment:
    """Represents a single vessel segment with its properties."""
    centre: np.ndarray  # Centreline coordinates (N, 2) as [row, col]
    angles: np.ndarray  # Orientation angles at each point
    side1: Optional[np.ndarray] = None  # Left edge coordinates
    side2: Optional[np.ndarray] = None  # Right edge coordinates
    diameters: Optional[np.ndarray] = None  # Diameter at each point
    im_profiles: Optional[np.ndarray] = None  # Image intensity profiles
    

@dataclass 
class SegmentationResult:
    """Container for segmentation results."""
    binary_mask: np.ndarray  # Binary segmentation mask
    wavelet_sum: np.ndarray  # Sum of wavelet coefficients
    vessels: List[VesselSegment] = field(default_factory=list)
    fov_mask: Optional[np.ndarray] = None


class IUWT:
    """
    Isotropic Undecimated Wavelet Transform (IUWT) implementation.
    
    Also known as the 'à trous' (with holes) wavelet transform.
    Uses B3-spline scaling function: h0 = [1, 4, 6, 4, 1] / 16
    """
    
    def __init__(self):
        # B3-spline derived filter (cubic B-spline)
        self.h0 = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    
    def _upsample_filter(self, h: np.ndarray, level: int) -> np.ndarray:
        """
        Upsample filter by inserting 2^(level-1) zeros between coefficients.
        
        For level j, insert 2^j - 1 zeros between adjacent coefficients of h0.
        """
        if level == 0:
            return h
        
        n_zeros = 2**level - 1
        new_len = len(h) + (len(h) - 1) * n_zeros
        h_up = np.zeros(new_len)
        h_up[::n_zeros + 1] = h
        return h_up
    
    def transform(self, image: np.ndarray, n_levels: int = 4) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute the IUWT of an image.
        
        Parameters
        ----------
        image : np.ndarray
            Input grayscale image
        n_levels : int
            Number of wavelet levels to compute
            
        Returns
        -------
        wavelet_coeffs : list of np.ndarray
            Wavelet coefficients w1, w2, ..., w_n
        scaling_coeffs : np.ndarray
            Final scaling coefficients c_n (smooth residual)
        """
        image = image.astype(np.float64)
        
        wavelet_coeffs = []
        c_prev = image.copy()
        
        for j in range(n_levels):
            # Get upsampled filter for this level
            h_j = self._upsample_filter(self.h0, j)
            
            # Apply separable filtering (2D convolution using 1D filters)
            # First along rows, then along columns
            c_next = ndimage.convolve1d(c_prev, h_j, axis=0, mode='reflect')
            c_next = ndimage.convolve1d(c_next, h_j, axis=1, mode='reflect')
            
            # Wavelet coefficients are the difference
            w_j = c_prev - c_next
            wavelet_coeffs.append(w_j)
            
            c_prev = c_next
        
        return wavelet_coeffs, c_prev
    
    def reconstruct(self, wavelet_coeffs: List[np.ndarray], 
                   scaling_coeffs: np.ndarray) -> np.ndarray:
        """
        Reconstruct image from wavelet and scaling coefficients.
        
        f = c_n + sum(w_j) for j = 1 to n
        """
        return scaling_coeffs + sum(wavelet_coeffs)


class VesselSegmenter:
    """
    Retinal vessel segmentation using IUWT.
    """
    
    def __init__(self, 
                 dark_vessels: bool = True,
                 wavelet_levels: List[int] = [2, 3],
                 threshold_percent: float = 0.20,
                 min_object_size_percent: float = 0.05,
                 fill_hole_size_percent: float = 0.05,
                 use_inpainting: bool = False):
        """
        Initialize the vessel segmenter.
        
        Parameters
        ----------
        dark_vessels : bool
            True if vessels are darker than background (fundus images),
            False if brighter (fluorescein angiograms)
        wavelet_levels : list of int
            Which wavelet levels to sum for segmentation (1-indexed)
        threshold_percent : float
            Percentage of pixels to identify as vessels (0.0 to 1.0)
        min_object_size_percent : float
            Minimum object size as percentage of FOV to keep
        fill_hole_size_percent : float
            Maximum hole size as percentage of FOV to fill
        use_inpainting : bool
            Whether to inpaint outside FOV before wavelet transform
        """
        self.dark_vessels = dark_vessels
        self.wavelet_levels = wavelet_levels
        self.threshold_percent = threshold_percent
        self.min_object_size_percent = min_object_size_percent
        self.fill_hole_size_percent = fill_hole_size_percent
        self.use_inpainting = use_inpainting
        self.iuwt = IUWT()
    
    def create_fov_mask(self, image: np.ndarray, 
                        threshold: int = 20,
                        erode_size: int = 3) -> np.ndarray:
        """
        Create a field of view (FOV) mask by thresholding.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color)
        threshold : int
            Threshold value for mask creation
        erode_size : int
            Size of erosion structuring element
            
        Returns
        -------
        mask : np.ndarray
            Binary FOV mask
        """
        if len(image.shape) == 3:
            # Use red channel for color images (best for FOV detection)
            gray = image[:, :, 0]
        else:
            gray = image
            
        mask = gray > threshold
        
        if erode_size > 0:
            kernel = np.ones((erode_size, erode_size), dtype=np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        
        return mask
    
    def _distance_inpainting(self, image: np.ndarray, 
                            mask: np.ndarray) -> np.ndarray:
        """
        Simple inpainting using distance transform.
        Replace pixels outside mask with nearest inside pixel values.
        """
        if mask is None or mask.all():
            return image
        
        # Find nearest inside pixel for each outside pixel
        dist, indices = distance_transform_edt(~mask, return_indices=True)
        
        # Replace outside pixels
        result = image.copy()
        outside = ~mask
        result[outside] = image[indices[0][outside], indices[1][outside]]
        
        return result
    
    def _percentage_threshold(self, wavelet_sum: np.ndarray,
                             mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Threshold wavelet coefficients based on percentage.
        """
        if mask is not None:
            values = wavelet_sum[mask]
        else:
            values = wavelet_sum.ravel()
        
        if self.dark_vessels:
            # Vessels are darker, look for lowest values
            thresh = np.percentile(values, self.threshold_percent * 100)
            binary = wavelet_sum <= thresh
        else:
            # Vessels are brighter, look for highest values
            thresh = np.percentile(values, (1 - self.threshold_percent) * 100)
            binary = wavelet_sum >= thresh
        
        # Apply mask if provided
        if mask is not None:
            binary = binary & mask
            
        return binary
    
    def _clean_binary_image(self, binary: np.ndarray,
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Clean binary image by removing small objects and filling holes.
        """
        # Calculate pixel counts for size thresholds
        if mask is not None:
            n_pixels = np.sum(mask)
        else:
            n_pixels = binary.size
        
        min_size = int(n_pixels * self.min_object_size_percent)
        fill_size = int(n_pixels * self.fill_hole_size_percent)
        
        # Remove small objects
        binary = remove_small_objects(binary, min_size=max(min_size, 1))
        
        # Fill small holes
        if fill_size > 0:
            # Label the holes (background connected components)
            inverted = ~binary
            if mask is not None:
                inverted = inverted & mask
            labeled_holes, n_holes = label(inverted)
            
            for i in range(1, n_holes + 1):
                hole_mask = labeled_holes == i
                if np.sum(hole_mask) <= fill_size:
                    binary = binary | hole_mask
        
        return binary
    
    def segment(self, image: np.ndarray,
               fov_mask: Optional[np.ndarray] = None) -> SegmentationResult:
        """
        Segment vessels in a retinal image.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (grayscale or color)
        fov_mask : np.ndarray, optional
            Field of view mask
            
        Returns
        -------
        result : SegmentationResult
            Segmentation results including binary mask and wavelet sum
        """
        # Extract green channel if color (best contrast for vessels)
        if len(image.shape) == 3:
            gray = image[:, :, 1].astype(np.float64)  # Green channel
        else:
            gray = image.astype(np.float64)
        
        # Apply inpainting if requested
        if self.use_inpainting and fov_mask is not None:
            gray = self._distance_inpainting(gray, fov_mask)
        
        # Compute IUWT
        max_level = max(self.wavelet_levels)
        wavelet_coeffs, _ = self.iuwt.transform(gray, max_level)
        
        # Sum selected wavelet levels (convert to 0-indexed)
        wavelet_sum = np.zeros_like(gray)
        for level in self.wavelet_levels:
            wavelet_sum += wavelet_coeffs[level - 1]
        
        # Threshold
        binary = self._percentage_threshold(wavelet_sum, fov_mask)
        
        # Clean
        binary = self._clean_binary_image(binary, fov_mask)
        
        return SegmentationResult(
            binary_mask=binary,
            wavelet_sum=wavelet_sum,
            fov_mask=fov_mask
        )


class CentrelineExtractor:
    """
    Extract and refine vessel centrelines from binary segmentation.
    """
    
    def __init__(self,
                 spur_length: int = 10,
                 min_length: int = 10,
                 spline_spacing: int = 10,
                 remove_extreme: bool = True,
                 clear_branches_dist: bool = True):
        """
        Initialize centreline extractor.
        
        Parameters
        ----------
        spur_length : int
            Length of spurs to remove from thinned skeleton
        min_length : int
            Minimum centreline length in pixels
        spline_spacing : int
            Approximate spacing between spline pieces
        remove_extreme : bool
            Remove segments where diameter estimate exceeds length
        clear_branches_dist : bool
            Use distance transform to clear pixels near branches
        """
        self.spur_length = spur_length
        self.min_length = min_length
        self.spline_spacing = spline_spacing
        self.remove_extreme = remove_extreme
        self.clear_branches_dist = clear_branches_dist
    
    def _find_neighbors(self, skeleton: np.ndarray, 
                       point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find 8-connected neighbors that are part of skeleton."""
        r, c = point
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (0 <= nr < skeleton.shape[0] and 
                    0 <= nc < skeleton.shape[1] and
                    skeleton[nr, nc]):
                    neighbors.append((nr, nc))
        return neighbors
    
    def _count_neighbors(self, skeleton: np.ndarray) -> np.ndarray:
        """Count 8-connected neighbors for each pixel."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        return cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    
    def _remove_spurs(self, skeleton: np.ndarray) -> np.ndarray:
        """Remove short spurs from skeleton."""
        skeleton = skeleton.copy()
        
        for _ in range(self.spur_length):
            # Count neighbors
            neighbor_count = self._count_neighbors(skeleton)
            
            # Find endpoints (1 neighbor)
            endpoints = (neighbor_count == 1) & skeleton
            
            if not np.any(endpoints):
                break
            
            # Remove endpoints
            skeleton[endpoints] = False
        
        return skeleton
    
    def _remove_branch_points(self, skeleton: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove branch points from skeleton.
        
        Returns
        -------
        skeleton : np.ndarray
            Skeleton with branch points removed
        branch_points : np.ndarray
            Binary image of branch point locations
        """
        skeleton = skeleton.copy()
        neighbor_count = self._count_neighbors(skeleton)
        
        # Branch points have > 2 neighbors
        branch_points = (neighbor_count > 2) & skeleton
        skeleton[branch_points] = False
        
        return skeleton, branch_points
    
    def _trace_segment(self, skeleton: np.ndarray,
                      start: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Trace a connected segment from a starting point."""
        segment = [start]
        visited = {start}
        skeleton_copy = skeleton.copy()
        skeleton_copy[start] = False
        
        # Trace in both directions from start
        for direction in [0, 1]:
            current = start
            while True:
                neighbors = self._find_neighbors(skeleton_copy, current)
                unvisited = [n for n in neighbors if n not in visited]
                
                if not unvisited:
                    break
                
                # Take the first unvisited neighbor
                next_point = unvisited[0]
                visited.add(next_point)
                skeleton_copy[next_point] = False
                
                if direction == 0:
                    segment.append(next_point)
                else:
                    segment.insert(0, next_point)
                
                current = next_point
        
        return segment
    
    def _extract_segments(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """Extract all connected segments from skeleton."""
        # Label connected components
        labeled, n_labels = label(skeleton)
        
        segments = []
        for i in range(1, n_labels + 1):
            # Get points in this component
            points = np.array(np.where(labeled == i)).T  # (N, 2)
            
            if len(points) < self.min_length:
                continue
            
            # Find an endpoint or any point to start tracing
            component_mask = labeled == i
            neighbor_count = self._count_neighbors(component_mask)
            endpoints = np.where((neighbor_count == 1) & component_mask)
            
            if len(endpoints[0]) > 0:
                start = (endpoints[0][0], endpoints[1][0])
            else:
                start = (points[0, 0], points[0, 1])
            
            # Trace the segment
            segment_points = self._trace_segment(component_mask, start)
            
            if len(segment_points) >= self.min_length:
                segments.append(np.array(segment_points))
        
        return segments
    
    def _fit_spline(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a cubic spline to centreline points.
        
        Uses centripetal parameterization for better results with curves.
        
        Returns
        -------
        smoothed_points : np.ndarray
            Smoothed centreline coordinates
        angles : np.ndarray
            Orientation angles at each point
        """
        if len(points) < 4:
            # Not enough points for spline, use original
            angles = self._compute_angles_simple(points)
            return points, angles
        
        # Centripetal parameterization
        diffs = np.diff(points, axis=0)
        chord_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        chord_lengths = np.sqrt(chord_lengths)  # Centripetal: use sqrt of chord length
        
        # Cumulative parameter
        t = np.zeros(len(points))
        t[1:] = np.cumsum(chord_lengths)
        t = t / t[-1]  # Normalize to [0, 1]
        
        # Determine number of spline pieces
        total_length = np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        n_pieces = max(1, int(total_length / self.spline_spacing))
        k = min(3, len(points) - 1)  # Spline degree
        
        try:
            # Fit spline
            tck, u = splprep([points[:, 0], points[:, 1]], u=t, k=k, 
                            s=len(points) * 0.5)
            
            # Evaluate at same parameter values
            smoothed = np.array(splev(t, tck)).T
            
            # Compute derivatives for angles
            derivs = np.array(splev(t, tck, der=1)).T
            angles = np.arctan2(derivs[:, 1], derivs[:, 0])
            
        except Exception:
            # Fall back to simple angle computation
            smoothed = points.copy()
            angles = self._compute_angles_simple(points)
        
        return smoothed, angles
    
    def _compute_angles_simple(self, points: np.ndarray) -> np.ndarray:
        """Compute angles using finite differences."""
        n = len(points)
        angles = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                diff = points[1] - points[0]
            elif i == n - 1:
                diff = points[-1] - points[-2]
            else:
                diff = points[i + 1] - points[i - 1]
            
            angles[i] = np.arctan2(diff[1], diff[0])
        
        return angles
    
    def extract(self, binary: np.ndarray) -> List[VesselSegment]:
        """
        Extract vessel centrelines from binary segmentation.
        
        Parameters
        ----------
        binary : np.ndarray
            Binary vessel segmentation
            
        Returns
        -------
        vessels : list of VesselSegment
            Extracted vessel segments with centrelines and angles
        """
        # Compute distance transform for diameter estimation
        dist_transform = distance_transform_edt(binary)
        
        # Skeletonize
        skeleton = skeletonize(binary)
        
        # Remove spurs
        skeleton = self._remove_spurs(skeleton)
        
        # Remove branch points
        skeleton, branch_points = self._remove_branch_points(skeleton)
        
        # Optionally clear pixels near branches using distance transform
        if self.clear_branches_dist and np.any(branch_points):
            branch_dist = distance_transform_edt(~branch_points)
            skeleton = skeleton & (branch_dist > 2)
        
        # Extract segments
        segment_points = self._extract_segments(skeleton)
        
        vessels = []
        for points in segment_points:
            # Estimate diameter from distance transform
            diameters_est = np.array([dist_transform[p[0], p[1]] * 2 
                                     for p in points])
            max_diameter = np.max(diameters_est)
            
            # Skip if segment is too short relative to diameter
            if self.remove_extreme and len(points) < max_diameter:
                continue
            
            # Fit spline and get angles
            smoothed, angles = self._fit_spline(points)
            
            vessel = VesselSegment(
                centre=smoothed,
                angles=angles,
                diameters=diameters_est
            )
            vessels.append(vessel)
        
        return vessels


class ProfileGenerator:
    """
    Generate image intensity profiles perpendicular to vessel centrelines.
    """
    
    def __init__(self, profile_width: Optional[int] = None):
        """
        Initialize profile generator.
        
        Parameters
        ----------
        profile_width : int, optional
            Width of profiles. If None, auto-calculated from vessel diameter.
        """
        self.profile_width = profile_width
    
    def generate(self, image: np.ndarray, 
                vessel: VesselSegment,
                binary: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate image profiles perpendicular to vessel centreline.
        
        Parameters
        ----------
        image : np.ndarray
            Grayscale image
        vessel : VesselSegment
            Vessel segment with centreline and angles
        binary : np.ndarray, optional
            Binary segmentation for width estimation
            
        Returns
        -------
        profiles : np.ndarray
            2D array of profiles (n_points x profile_width)
        """
        centre = vessel.centre
        angles = vessel.angles
        n_points = len(centre)
        
        # Determine profile width
        if self.profile_width is not None:
            width = self.profile_width
        elif vessel.diameters is not None:
            # Use 4x maximum diameter estimate
            width = int(np.max(vessel.diameters) * 4)
        else:
            width = 31  # Default
        
        # Ensure odd width
        if width % 2 == 0:
            width += 1
        
        half_width = width // 2
        
        # Generate profiles
        profiles = np.zeros((n_points, width))
        
        for i in range(n_points):
            row, col = centre[i]
            angle = angles[i]
            
            # Perpendicular direction
            perp_angle = angle + np.pi / 2
            
            # Sample points along perpendicular
            offsets = np.arange(-half_width, half_width + 1)
            sample_rows = row + offsets * np.sin(perp_angle)
            sample_cols = col + offsets * np.cos(perp_angle)
            
            # Bilinear interpolation
            for j, (sr, sc) in enumerate(zip(sample_rows, sample_cols)):
                if (0 <= sr < image.shape[0] - 1 and 
                    0 <= sc < image.shape[1] - 1):
                    # Bilinear interpolation
                    r0, c0 = int(sr), int(sc)
                    r1, c1 = r0 + 1, c0 + 1
                    dr, dc = sr - r0, sc - c0
                    
                    profiles[i, j] = (
                        image[r0, c0] * (1 - dr) * (1 - dc) +
                        image[r1, c0] * dr * (1 - dc) +
                        image[r0, c1] * (1 - dr) * dc +
                        image[r1, c1] * dr * dc
                    )
                else:
                    profiles[i, j] = np.nan
        
        vessel.im_profiles = profiles
        return profiles


class EdgeDetector:
    """
    Detect vessel edges using zero-crossings of the second derivative.
    """
    
    def __init__(self,
                 smooth_parallel: float = 1.0,
                 smooth_perpendicular: float = 0.1,
                 enforce_connectivity: bool = True):
        """
        Initialize edge detector.
        
        Parameters
        ----------
        smooth_parallel : float
            Smoothing scale parallel to vessel (multiplies sqrt(width))
        smooth_perpendicular : float
            Smoothing scale perpendicular to vessel (multiplies sqrt(width))
        enforce_connectivity : bool
            Enforce edge connectivity constraint
        """
        self.smooth_parallel = smooth_parallel
        self.smooth_perpendicular = smooth_perpendicular
        self.enforce_connectivity = enforce_connectivity
    
    def _estimate_width_from_profiles(self, profiles: np.ndarray,
                                      binary_profiles: Optional[np.ndarray] = None) -> float:
        """Estimate vessel width from profiles."""
        if binary_profiles is not None:
            # Sum of vessel pixels per profile
            widths = np.sum(binary_profiles, axis=1)
            return np.median(widths[widths > 0])
        else:
            # Use gradient-based estimation
            mean_profile = np.nanmean(profiles, axis=0)
            gradient = np.gradient(mean_profile)
            
            center = len(mean_profile) // 2
            
            # Find max gradient (left edge) and min gradient (right edge)
            left_region = gradient[:center]
            right_region = gradient[center:]
            
            left_edge = np.argmax(left_region)
            right_edge = center + np.argmin(right_region)
            
            return right_edge - left_edge
    
    def _find_zero_crossings(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find zero crossings in signal.
        
        Returns positions and types (+1 for negative-to-positive, -1 for positive-to-negative)
        """
        signs = np.sign(signal)
        sign_changes = np.diff(signs)
        
        # Positive-to-negative (falling edge): sign_changes == -2
        # Negative-to-positive (rising edge): sign_changes == 2
        
        rising = np.where(sign_changes > 0)[0]
        falling = np.where(sign_changes < 0)[0]
        
        # Sub-pixel interpolation
        rising_positions = []
        for idx in rising:
            if signal[idx] != 0:
                frac = -signal[idx] / (signal[idx + 1] - signal[idx])
                rising_positions.append(idx + frac)
        
        falling_positions = []
        for idx in falling:
            if signal[idx] != 0:
                frac = -signal[idx] / (signal[idx + 1] - signal[idx])
                falling_positions.append(idx + frac)
        
        return np.array(rising_positions), np.array(falling_positions)
    
    def _connect_edges(self, edge_positions: List[np.ndarray],
                      predicted_col: float,
                      width_estimate: float) -> np.ndarray:
        """
        Connect edge detections into continuous trails.
        """
        n_profiles = len(edge_positions)
        connected = np.full(n_profiles, np.nan)
        
        # Accept edges within 1/3 diameter of predicted location
        tolerance = width_estimate / 3
        
        for i, positions in enumerate(edge_positions):
            if len(positions) == 0:
                continue
            
            # Filter positions within tolerance
            valid = positions[np.abs(positions - predicted_col) < tolerance]
            
            if len(valid) > 0:
                # Take closest to predicted
                idx = np.argmin(np.abs(valid - predicted_col))
                connected[i] = valid[idx]
        
        if self.enforce_connectivity:
            # Find longest connected segment
            connected = self._longest_connected(connected)
        
        return connected
    
    def _longest_connected(self, edges: np.ndarray) -> np.ndarray:
        """Keep only the longest connected segment of edges."""
        valid = ~np.isnan(edges)
        
        if not np.any(valid):
            return edges
        
        # Find connected runs
        labeled, n_labels = label(valid)
        
        if n_labels == 0:
            return edges
        
        # Find longest run
        longest_label = 0
        longest_length = 0
        for i in range(1, n_labels + 1):
            length = np.sum(labeled == i)
            if length > longest_length:
                longest_length = length
                longest_label = i
        
        # Keep only longest
        result = np.full_like(edges, np.nan)
        mask = labeled == longest_label
        result[mask] = edges[mask]
        
        return result
    
    def detect(self, vessel: VesselSegment, 
              profiles: Optional[np.ndarray] = None) -> VesselSegment:
        """
        Detect vessel edges from image profiles.
        
        Parameters
        ----------
        vessel : VesselSegment
            Vessel with profiles already computed
        profiles : np.ndarray, optional
            Override vessel's stored profiles
            
        Returns
        -------
        vessel : VesselSegment
            Updated vessel with edge locations and diameters
        """
        if profiles is None:
            profiles = vessel.im_profiles
        
        if profiles is None:
            raise ValueError("No profiles available")
        
        n_profiles, profile_width = profiles.shape
        center_col = profile_width // 2
        
        # Step 1: Estimate vessel width
        width_estimate = self._estimate_width_from_profiles(profiles)
        
        # Step 2: Find predicted edge locations from mean profile
        mean_profile = np.nanmean(profiles, axis=0)
        gradient = np.gradient(mean_profile)
        
        # Search within one diameter of center
        search_range = int(width_estimate)
        left_start = max(0, center_col - search_range)
        left_end = center_col
        right_start = center_col
        right_end = min(profile_width, center_col + search_range)
        
        left_gradient = gradient[left_start:left_end]
        right_gradient = gradient[right_start:right_end]
        
        pred_left = left_start + np.argmax(left_gradient)
        pred_right = right_start + np.argmin(right_gradient)
        
        # Refine width estimate
        width_estimate = pred_right - pred_left
        
        # Step 3: Apply anisotropic Gaussian smoothing
        sigma_h = np.sqrt(self.smooth_perpendicular * width_estimate)
        sigma_v = np.sqrt(self.smooth_parallel * width_estimate)
        
        smoothed = ndimage.gaussian_filter(profiles, sigma=(sigma_v, sigma_h))
        
        # Step 4: Compute second derivative perpendicular to vessel
        second_deriv = np.gradient(np.gradient(smoothed, axis=1), axis=1)
        
        # Step 5: Find zero crossings for each profile
        left_edges_list = []
        right_edges_list = []
        
        for i in range(n_profiles):
            rising, falling = self._find_zero_crossings(second_deriv[i])
            
            # Left edge: positive-to-negative (falling in 2nd deriv)
            # Right edge: negative-to-positive (rising in 2nd deriv)
            left_candidates = falling[falling < center_col]
            right_candidates = rising[rising > center_col]
            
            left_edges_list.append(left_candidates)
            right_edges_list.append(right_candidates)
        
        # Connect edges
        left_edges = self._connect_edges(left_edges_list, pred_left, width_estimate)
        right_edges = self._connect_edges(right_edges_list, pred_right, width_estimate)
        
        # Compute diameters
        diameters = right_edges - left_edges
        
        # Convert edge positions to image coordinates
        side1 = np.zeros((n_profiles, 2))
        side2 = np.zeros((n_profiles, 2))
        
        for i in range(n_profiles):
            if np.isnan(left_edges[i]) or np.isnan(right_edges[i]):
                side1[i] = [np.nan, np.nan]
                side2[i] = [np.nan, np.nan]
            else:
                row, col = vessel.centre[i]
                angle = vessel.angles[i]
                perp_angle = angle + np.pi / 2
                
                left_offset = left_edges[i] - center_col
                right_offset = right_edges[i] - center_col
                
                side1[i, 0] = row + left_offset * np.sin(perp_angle)
                side1[i, 1] = col + left_offset * np.cos(perp_angle)
                side2[i, 0] = row + right_offset * np.sin(perp_angle)
                side2[i, 1] = col + right_offset * np.cos(perp_angle)
        
        vessel.side1 = side1
        vessel.side2 = side2
        vessel.diameters = diameters
        
        return vessel


class RetinalVesselAnalyzer:
    """
    Complete retinal vessel analysis pipeline.
    
    Combines segmentation, centreline extraction, and edge detection.
    """
    
    def __init__(self,
                 # Segmentation parameters
                 dark_vessels: bool = True,
                 wavelet_levels: List[int] = [2, 3],
                 threshold_percent: float = 0.20,
                 min_object_size_percent: float = 0.05,
                 fill_hole_size_percent: float = 0.05,
                 # Centreline parameters
                 spur_length: int = 10,
                 min_segment_length: int = 10,
                 spline_spacing: int = 10,
                 # Edge detection parameters
                 smooth_parallel: float = 1.0,
                 smooth_perpendicular: float = 0.1,
                 enforce_connectivity: bool = True):
        """
        Initialize the analyzer with all parameters.
        
        See component classes for parameter descriptions.
        """
        self.segmenter = VesselSegmenter(
            dark_vessels=dark_vessels,
            wavelet_levels=wavelet_levels,
            threshold_percent=threshold_percent,
            min_object_size_percent=min_object_size_percent,
            fill_hole_size_percent=fill_hole_size_percent
        )
        
        self.centreline_extractor = CentrelineExtractor(
            spur_length=spur_length,
            min_length=min_segment_length,
            spline_spacing=spline_spacing
        )
        
        self.profile_generator = ProfileGenerator()
        
        self.edge_detector = EdgeDetector(
            smooth_parallel=smooth_parallel,
            smooth_perpendicular=smooth_perpendicular,
            enforce_connectivity=enforce_connectivity
        )
    
    def analyze(self, image: np.ndarray,
               fov_mask: Optional[np.ndarray] = None,
               create_fov_mask: bool = True) -> SegmentationResult:
        """
        Perform complete vessel analysis.
        
        Parameters
        ----------
        image : np.ndarray
            Input retinal image (color or grayscale)
        fov_mask : np.ndarray, optional
            Field of view mask
        create_fov_mask : bool
            Create FOV mask automatically if not provided
            
        Returns
        -------
        result : SegmentationResult
            Complete analysis results
        """
        # Create FOV mask if needed
        if fov_mask is None and create_fov_mask:
            fov_mask = self.segmenter.create_fov_mask(image)
        
        # Segment vessels
        result = self.segmenter.segment(image, fov_mask)
        
        # Extract centrelines
        vessels = self.centreline_extractor.extract(result.binary_mask)
        
        # Get grayscale image for profiles
        if len(image.shape) == 3:
            gray = image[:, :, 1].astype(np.float64)  # Green channel
        else:
            gray = image.astype(np.float64)
        
        # Process each vessel
        for vessel in vessels:
            # Generate profiles
            self.profile_generator.generate(gray, vessel)
            
            # Detect edges
            try:
                self.edge_detector.detect(vessel)
            except Exception:
                # Edge detection may fail for some vessels
                pass
        
        result.vessels = vessels
        return result
    
    def get_all_diameters(self, result: SegmentationResult) -> np.ndarray:
        """
        Get all diameter measurements from analysis result.
        
        Returns
        -------
        diameters : np.ndarray
            All diameter measurements concatenated
        """
        all_diameters = []
        for vessel in result.vessels:
            if vessel.diameters is not None:
                valid = vessel.diameters[~np.isnan(vessel.diameters)]
                all_diameters.extend(valid)
        return np.array(all_diameters)
    
    def get_statistics(self, result: SegmentationResult) -> Dict:
        """
        Get summary statistics from analysis.
        
        Returns
        -------
        stats : dict
            Dictionary of statistics
        """
        diameters = self.get_all_diameters(result)
        
        n_vessels = len(result.vessels)
        n_measurements = len(diameters)
        
        stats = {
            'num_vessels': n_vessels,
            'num_measurements': n_measurements,
            'vessel_pixel_ratio': np.mean(result.binary_mask) if result.fov_mask is None 
                                  else np.sum(result.binary_mask & result.fov_mask) / np.sum(result.fov_mask),
        }
        
        if n_measurements > 0:
            stats.update({
                'mean_diameter': np.mean(diameters),
                'std_diameter': np.std(diameters),
                'min_diameter': np.min(diameters),
                'max_diameter': np.max(diameters),
                'median_diameter': np.median(diameters),
            })
        
        return stats


# Convenience function for quick analysis
def analyze_retinal_image(image: np.ndarray,
                         dark_vessels: bool = True,
                         wavelet_levels: List[int] = [2, 3],
                         threshold_percent: float = 0.20) -> SegmentationResult:
    """
    Convenience function for quick retinal image analysis.
    
    Parameters
    ----------
    image : np.ndarray
        Input retinal image
    dark_vessels : bool
        True for fundus images, False for fluorescein angiograms
    wavelet_levels : list of int
        Wavelet levels to use (adjust for image resolution)
    threshold_percent : float
        Percentage of pixels to classify as vessels
        
    Returns
    -------
    result : SegmentationResult
        Analysis results
    """
    analyzer = RetinalVesselAnalyzer(
        dark_vessels=dark_vessels,
        wavelet_levels=wavelet_levels,
        threshold_percent=threshold_percent
    )
    return analyzer.analyze(image)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vessel_segmentation.py <image_path>")
        print("\nThis library provides retinal vessel segmentation and measurement.")
        print("See the docstrings for detailed API documentation.")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Analyze
    print("Analyzing image...")
    result = analyze_retinal_image(image)
    
    # Print statistics
    analyzer = RetinalVesselAnalyzer()
    stats = analyzer.get_statistics(result)
    
    print("\nAnalysis Results:")
    print(f"  Number of vessel segments: {stats['num_vessels']}")
    print(f"  Number of diameter measurements: {stats['num_measurements']}")
    print(f"  Vessel pixel ratio: {stats['vessel_pixel_ratio']:.4f}")
    
    if stats['num_measurements'] > 0:
        print(f"  Mean diameter: {stats['mean_diameter']:.2f} pixels")
        print(f"  Std diameter: {stats['std_diameter']:.2f} pixels")
        print(f"  Min diameter: {stats['min_diameter']:.2f} pixels")
        print(f"  Max diameter: {stats['max_diameter']:.2f} pixels")
