"""
Visualization utilities for retinal vessel segmentation.

Provides functions to visualize:
- Segmentation results
- Vessel centrelines and edges
- Image profiles
- Wavelet coefficients
- Diameter measurements
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cv2
from typing import Optional, Tuple, List, Union

from .core import SegmentationResult, VesselSegment


def show_segmentation(image: np.ndarray,
                     result: SegmentationResult,
                     alpha: float = 0.5,
                     figsize: Tuple[int, int] = (12, 6),
                     save_path: Optional[str] = None):
    """
    Display original image alongside segmentation.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    result : SegmentationResult
        Segmentation results
    alpha : float
        Overlay transparency
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    if len(image.shape) == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Binary segmentation
    axes[1].imshow(result.binary_mask, cmap='gray')
    axes[1].set_title('Binary Segmentation')
    axes[1].axis('off')
    
    # Overlay
    if len(image.shape) == 3:
        overlay = image.copy()
    else:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create colored mask
    mask_overlay = np.zeros_like(overlay)
    mask_overlay[result.binary_mask, 0] = 255  # Red channel
    
    blended = cv2.addWeighted(overlay, 1 - alpha, mask_overlay, alpha, 0)
    axes[2].imshow(blended)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def show_wavelet_levels(wavelet_coeffs: List[np.ndarray],
                       scaling_coeffs: np.ndarray,
                       figsize: Optional[Tuple[int, int]] = None,
                       save_path: Optional[str] = None):
    """
    Display wavelet coefficients at each level.
    
    Parameters
    ----------
    wavelet_coeffs : list of np.ndarray
        Wavelet coefficients from IUWT
    scaling_coeffs : np.ndarray
        Final scaling coefficients
    figsize : tuple, optional
        Figure size
    save_path : str, optional
        Path to save figure
    """
    n_levels = len(wavelet_coeffs)
    n_cols = min(3, n_levels + 1)
    n_rows = (n_levels + 1 + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i, w in enumerate(wavelet_coeffs):
        # Normalize for display
        vmax = np.max(np.abs(w))
        axes[i].imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Wavelet Level {i + 1}')
        axes[i].axis('off')
    
    # Scaling coefficients
    axes[n_levels].imshow(scaling_coeffs, cmap='gray')
    axes[n_levels].set_title('Smooth Residual')
    axes[n_levels].axis('off')
    
    # Hide unused axes
    for i in range(n_levels + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def show_centrelines(image: np.ndarray,
                    vessels: List[VesselSegment],
                    show_edges: bool = True,
                    color_by_diameter: bool = False,
                    figsize: Tuple[int, int] = (10, 10),
                    save_path: Optional[str] = None):
    """
    Display vessel centrelines on image.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    vessels : list of VesselSegment
        Vessel segments
    show_edges : bool
        Whether to show detected edges
    color_by_diameter : bool
        Color centrelines by diameter
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    if len(image.shape) == 3:
        ax.imshow(image)
    else:
        ax.imshow(image, cmap='gray')
    
    # Collect all diameters for colormap normalization
    if color_by_diameter:
        all_diameters = []
        for v in vessels:
            if v.diameters is not None:
                all_diameters.extend(v.diameters[~np.isnan(v.diameters)])
        
        if all_diameters:
            norm = Normalize(vmin=np.min(all_diameters), vmax=np.max(all_diameters))
            cmap = plt.cm.viridis
        else:
            color_by_diameter = False
    
    for i, vessel in enumerate(vessels):
        centre = vessel.centre
        
        if color_by_diameter and vessel.diameters is not None:
            # Color by diameter
            for j in range(len(centre) - 1):
                if not np.isnan(vessel.diameters[j]):
                    color = cmap(norm(vessel.diameters[j]))
                    ax.plot(centre[j:j+2, 1], centre[j:j+2, 0], 
                           color=color, linewidth=1)
        else:
            ax.plot(centre[:, 1], centre[:, 0], 'b-', linewidth=1, alpha=0.7)
        
        # Show edges
        if show_edges and vessel.side1 is not None and vessel.side2 is not None:
            valid = ~np.isnan(vessel.side1[:, 0])
            ax.plot(vessel.side1[valid, 1], vessel.side1[valid, 0], 
                   'r-', linewidth=0.5, alpha=0.5)
            ax.plot(vessel.side2[valid, 1], vessel.side2[valid, 0], 
                   'r-', linewidth=0.5, alpha=0.5)
    
    ax.set_title(f'Vessel Centrelines ({len(vessels)} segments)')
    ax.axis('off')
    
    if color_by_diameter and all_diameters:
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Diameter (pixels)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def show_vessel_profile(vessel: VesselSegment,
                       profile_idx: Optional[int] = None,
                       figsize: Tuple[int, int] = (12, 4),
                       save_path: Optional[str] = None):
    """
    Display vessel intensity profile(s).
    
    Parameters
    ----------
    vessel : VesselSegment
        Vessel segment with profiles
    profile_idx : int, optional
        Specific profile index to show. If None, shows mean profile.
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    if vessel.im_profiles is None:
        print("No profiles available for this vessel")
        return
    
    profiles = vessel.im_profiles
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Show straightened vessel image
    axes[0].imshow(profiles, aspect='auto', cmap='gray')
    axes[0].set_xlabel('Profile position')
    axes[0].set_ylabel('Along vessel')
    axes[0].set_title('Straightened Vessel')
    
    # Mark edges if available
    if vessel.side1 is not None and vessel.side2 is not None:
        # Edge positions in profile coordinates
        center = profiles.shape[1] // 2
        
        # This is simplified - actual edge positions would need to be tracked
        axes[0].axvline(center, color='b', linestyle='--', alpha=0.5, label='Center')
    
    # Show profile
    if profile_idx is not None:
        profile = profiles[profile_idx]
        title = f'Profile at index {profile_idx}'
    else:
        profile = np.nanmean(profiles, axis=0)
        title = 'Mean Profile'
    
    x = np.arange(len(profile)) - len(profile) // 2
    axes[1].plot(x, profile, 'b-', linewidth=2)
    
    # Show gradient
    gradient = np.gradient(profile)
    axes[1].plot(x, gradient * 10, 'r--', alpha=0.5, label='Gradient (×10)')
    
    axes[1].axvline(0, color='k', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Distance from center (pixels)')
    axes[1].set_ylabel('Intensity')
    axes[1].set_title(title)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def show_diameter_histogram(result: SegmentationResult,
                           bins: int = 50,
                           figsize: Tuple[int, int] = (8, 5),
                           save_path: Optional[str] = None):
    """
    Display histogram of diameter measurements.
    
    Parameters
    ----------
    result : SegmentationResult
        Segmentation results with vessels
    bins : int
        Number of histogram bins
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    all_diameters = []
    for vessel in result.vessels:
        if vessel.diameters is not None:
            valid = vessel.diameters[~np.isnan(vessel.diameters)]
            all_diameters.extend(valid)
    
    if not all_diameters:
        print("No diameter measurements available")
        return
    
    diameters = np.array(all_diameters)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.hist(diameters, bins=bins, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(diameters), color='r', linestyle='--', 
               label=f'Mean: {np.mean(diameters):.2f}')
    ax.axvline(np.median(diameters), color='g', linestyle='--', 
               label=f'Median: {np.median(diameters):.2f}')
    
    ax.set_xlabel('Diameter (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Diameter Distribution (n={len(diameters)})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def show_processing_steps(image: np.ndarray,
                         result: SegmentationResult,
                         wavelet_coeffs: List[np.ndarray],
                         levels_used: List[int],
                         figsize: Tuple[int, int] = (16, 8),
                         save_path: Optional[str] = None):
    """
    Display the main processing steps of the algorithm.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    result : SegmentationResult
        Segmentation results
    wavelet_coeffs : list of np.ndarray
        Wavelet coefficients
    levels_used : list of int
        Which wavelet levels were used (1-indexed)
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()
    
    # 1. Original
    if len(image.shape) == 3:
        axes[0].imshow(image)
    else:
        axes[0].imshow(image, cmap='gray')
    axes[0].set_title('(A) Original')
    axes[0].axis('off')
    
    # 2. Green channel
    if len(image.shape) == 3:
        green = image[:, :, 1]
    else:
        green = image
    axes[1].imshow(green, cmap='gray')
    axes[1].set_title('(B) Green Channel')
    axes[1].axis('off')
    
    # 3. FOV Mask
    if result.fov_mask is not None:
        axes[2].imshow(result.fov_mask, cmap='gray')
    else:
        axes[2].imshow(np.ones_like(green), cmap='gray')
    axes[2].set_title('(C) FOV Mask')
    axes[2].axis('off')
    
    # 4. Wavelet sum
    vmax = np.max(np.abs(result.wavelet_sum))
    axes[3].imshow(result.wavelet_sum, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[3].set_title(f'(D) Wavelet Sum (levels {levels_used})')
    axes[3].axis('off')
    
    # 5. Thresholded
    axes[4].imshow(result.binary_mask, cmap='gray')
    axes[4].set_title('(E) Thresholded')
    axes[4].axis('off')
    
    # 6. Cleaned
    axes[5].imshow(result.binary_mask, cmap='gray')
    axes[5].set_title('(F) Cleaned')
    axes[5].axis('off')
    
    # 7. Centrelines
    from skimage.morphology import skeletonize
    skeleton = skeletonize(result.binary_mask)
    axes[6].imshow(skeleton, cmap='gray')
    axes[6].set_title('(G) Thinned')
    axes[6].axis('off')
    
    # 8. Final edges
    if len(image.shape) == 3:
        overlay = image.copy()
    else:
        overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    for vessel in result.vessels:
        if vessel.side1 is not None and vessel.side2 is not None:
            valid = ~np.isnan(vessel.side1[:, 0])
            for idx in np.where(valid)[0]:
                r1, c1 = int(vessel.side1[idx, 0]), int(vessel.side1[idx, 1])
                r2, c2 = int(vessel.side2[idx, 0]), int(vessel.side2[idx, 1])
                if 0 <= r1 < overlay.shape[0] and 0 <= c1 < overlay.shape[1]:
                    overlay[r1, c1] = [255, 0, 0]
                if 0 <= r2 < overlay.shape[0] and 0 <= c2 < overlay.shape[1]:
                    overlay[r2, c2] = [255, 0, 0]
    
    axes[7].imshow(overlay)
    axes[7].set_title('(J) Detected Edges')
    axes[7].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def create_edge_overlay(image: np.ndarray,
                       result: SegmentationResult,
                       edge_color: Tuple[int, int, int] = (255, 0, 0),
                       centerline_color: Tuple[int, int, int] = (0, 0, 255),
                       thickness: int = 1) -> np.ndarray:
    """
    Create an image with vessel edges overlaid.
    
    Parameters
    ----------
    image : np.ndarray
        Original image
    result : SegmentationResult
        Segmentation results
    edge_color : tuple
        RGB color for edges
    centerline_color : tuple
        RGB color for centrelines
    thickness : int
        Line thickness
        
    Returns
    -------
    overlay : np.ndarray
        Image with overlaid edges
    """
    if len(image.shape) == 3:
        overlay = image.copy()
    else:
        overlay = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
    for vessel in result.vessels:
        # Draw centreline
        centre = vessel.centre.astype(np.int32)
        for i in range(len(centre) - 1):
            pt1 = (centre[i, 1], centre[i, 0])
            pt2 = (centre[i + 1, 1], centre[i + 1, 0])
            cv2.line(overlay, pt1, pt2, centerline_color, thickness)
        
        # Draw edges
        if vessel.side1 is not None and vessel.side2 is not None:
            valid = ~np.isnan(vessel.side1[:, 0])
            side1 = vessel.side1[valid].astype(np.int32)
            side2 = vessel.side2[valid].astype(np.int32)
            
            for i in range(len(side1) - 1):
                pt1 = (side1[i, 1], side1[i, 0])
                pt2 = (side1[i + 1, 1], side1[i + 1, 0])
                cv2.line(overlay, pt1, pt2, edge_color, thickness)
                
                pt1 = (side2[i, 1], side2[i, 0])
                pt2 = (side2[i + 1, 1], side2[i + 1, 0])
                cv2.line(overlay, pt1, pt2, edge_color, thickness)
    
    return overlay
