"""
Demo script for retinal vessel segmentation.

This script demonstrates the usage of the vessel segmentation library
on sample images or user-provided images.
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Handle imports whether running as script or from package
try:
    # First try importing as installed package
    from vessel_segmentation import (
        RetinalVesselAnalyzer,
        VesselSegmenter,
        CentrelineExtractor,
        IUWT,
        analyze_retinal_image
    )
    from vessel_segmentation import (
        show_segmentation,
        show_wavelet_levels,
        show_centrelines,
        show_diameter_histogram,
        show_processing_steps,
        create_edge_overlay
    )
except ImportError:
    try:
        # Try relative imports (when running from within package)
        from .core import (
            RetinalVesselAnalyzer,
            VesselSegmenter,
            CentrelineExtractor,
            IUWT,
            analyze_retinal_image
        )
        from .visualization import (
            show_segmentation,
            show_wavelet_levels,
            show_centrelines,
            show_diameter_histogram,
            show_processing_steps,
            create_edge_overlay
        )
    except ImportError:
        # Running as standalone script in same directory
        from core import (
            RetinalVesselAnalyzer,
            VesselSegmenter,
            CentrelineExtractor,
            IUWT,
            analyze_retinal_image
        )
        from visualization import (
            show_segmentation,
            show_wavelet_levels,
            show_centrelines,
            show_diameter_histogram,
            show_processing_steps,
            create_edge_overlay
        )


def create_test_image(size: int = 256) -> np.ndarray:
    """
    Create a simple test image with vessel-like structures.
    
    Parameters
    ----------
    size : int
        Image size
        
    Returns
    -------
    image : np.ndarray
        Synthetic test image
    """
    # Create background with slight gradient (like fundus image)
    y, x = np.ogrid[:size, :size]
    center = size // 2
    
    # Circular FOV
    r = np.sqrt((x - center)**2 + (y - center)**2)
    fov_mask = r < size * 0.45
    
    # Background
    background = 180 - 20 * r / (size * 0.5)
    background = np.clip(background, 100, 200)
    
    # Add noise
    noise = np.random.randn(size, size) * 5
    image = background + noise
    
    # Add vessel-like structures (darker lines)
    vessels = np.zeros((size, size))
    
    # Main horizontal vessel
    for i in range(size):
        y_pos = center + int(20 * np.sin(i * 0.02))
        width = 5 + int(3 * np.sin(i * 0.03))
        for dy in range(-width, width + 1):
            if 0 <= y_pos + dy < size:
                # Gaussian profile
                intensity = np.exp(-dy**2 / (2 * (width/2)**2))
                vessels[y_pos + dy, i] = max(vessels[y_pos + dy, i], intensity)
    
    # Branch
    for i in range(center, center + 80):
        y_pos = center + int((i - center) * 0.5 + 10 * np.sin((i - center) * 0.05))
        width = 3
        for dy in range(-width, width + 1):
            if 0 <= y_pos + dy < size:
                intensity = np.exp(-dy**2 / (2 * (width/2)**2))
                vessels[y_pos + dy, i] = max(vessels[y_pos + dy, i], intensity)
    
    # Subtract vessels (darker than background)
    image = image - 50 * vessels
    
    # Apply FOV mask
    image[~fov_mask] = 0
    
    # Normalize to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert to 3-channel (fundus-like)
    result = np.zeros((size, size, 3), dtype=np.uint8)
    result[:, :, 0] = np.clip(image * 0.8, 0, 255)  # Red
    result[:, :, 1] = image  # Green (best contrast)
    result[:, :, 2] = np.clip(image * 0.5, 0, 255)  # Blue
    
    return result


def run_demo(image_path: str = None, 
             output_dir: str = 'output',
             wavelet_levels: list = [2, 3],
             threshold: float = 0.20,
             show_plots: bool = True,
             save_plots: bool = True):
    """
    Run vessel segmentation demo.
    
    Parameters
    ----------
    image_path : str, optional
        Path to input image. If None, uses synthetic test image.
    output_dir : str
        Output directory for results
    wavelet_levels : list
        Wavelet levels to use
    threshold : float
        Segmentation threshold (percentage)
    show_plots : bool
        Show plots interactively
    save_plots : bool
        Save plots to output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or create image
    if image_path is not None:
        print(f"Loading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
    else:
        print("Creating synthetic test image...")
        image = create_test_image(512)
        image_name = "synthetic"
    
    print(f"Image shape: {image.shape}")
    
    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = RetinalVesselAnalyzer(
        dark_vessels=True,
        wavelet_levels=wavelet_levels,
        threshold_percent=threshold,
        min_object_size_percent=0.001,
        fill_hole_size_percent=0.001,
        spur_length=10,
        min_segment_length=10,
        spline_spacing=10,
        smooth_parallel=1.0,
        smooth_perpendicular=0.1
    )
    
    # Run analysis
    print("Running analysis...")
    result = analyzer.analyze(image)
    
    # Get statistics
    stats = analyzer.get_statistics(result)
    
    print("\n" + "="*50)
    print("ANALYSIS RESULTS")
    print("="*50)
    print(f"Number of vessel segments: {stats['num_vessels']}")
    print(f"Number of diameter measurements: {stats['num_measurements']}")
    print(f"Vessel pixel ratio: {stats['vessel_pixel_ratio']:.4f}")
    
    if stats['num_measurements'] > 0:
        print(f"\nDiameter Statistics (pixels):")
        print(f"  Mean: {stats['mean_diameter']:.2f}")
        print(f"  Std:  {stats['std_diameter']:.2f}")
        print(f"  Min:  {stats['min_diameter']:.2f}")
        print(f"  Max:  {stats['max_diameter']:.2f}")
        print(f"  Median: {stats['median_diameter']:.2f}")
    print("="*50)
    
    # Save binary mask
    mask_path = os.path.join(output_dir, f"{image_name}_segmentation.png")
    cv2.imwrite(mask_path, (result.binary_mask * 255).astype(np.uint8))
    print(f"\nSaved segmentation mask to: {mask_path}")
    
    # Create and save overlay
    overlay = create_edge_overlay(image, result)
    overlay_path = os.path.join(output_dir, f"{image_name}_overlay.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay to: {overlay_path}")
    
    # Generate visualizations
    if show_plots or save_plots:
        # Compute wavelet coefficients for visualization
        iuwt = IUWT()
        if len(image.shape) == 3:
            gray = image[:, :, 1].astype(np.float64)
        else:
            gray = image.astype(np.float64)
        wavelet_coeffs, scaling_coeffs = iuwt.transform(gray, max(wavelet_levels))
        
        # Segmentation overview
        save_path = os.path.join(output_dir, f"{image_name}_segmentation_overview.png") if save_plots else None
        if show_plots:
            show_segmentation(image, result, save_path=save_path)
        elif save_plots:
            fig, axes = plt.subplots(1, 3, figsize=(12, 6))
            if len(image.shape) == 3:
                axes[0].imshow(image)
            else:
                axes[0].imshow(image, cmap='gray')
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(result.binary_mask, cmap='gray')
            axes[1].set_title('Binary Segmentation')
            axes[1].axis('off')
            overlay = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            mask_overlay = np.zeros_like(overlay)
            mask_overlay[result.binary_mask, 0] = 255
            blended = cv2.addWeighted(overlay, 0.5, mask_overlay, 0.5, 0)
            axes[2].imshow(blended)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved segmentation overview to: {save_path}")
        
        # Wavelet levels
        save_path = os.path.join(output_dir, f"{image_name}_wavelet_levels.png") if save_plots else None
        if show_plots:
            show_wavelet_levels(wavelet_coeffs, scaling_coeffs, save_path=save_path)
        elif save_plots:
            n_levels = len(wavelet_coeffs)
            fig, axes = plt.subplots(1, n_levels + 1, figsize=(4 * (n_levels + 1), 4))
            for i, w in enumerate(wavelet_coeffs):
                vmax = np.max(np.abs(w))
                axes[i].imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                axes[i].set_title(f'Level {i + 1}')
                axes[i].axis('off')
            axes[-1].imshow(scaling_coeffs, cmap='gray')
            axes[-1].set_title('Residual')
            axes[-1].axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved wavelet levels to: {save_path}")
        
        # Centrelines
        save_path = os.path.join(output_dir, f"{image_name}_centrelines.png") if save_plots else None
        if show_plots:
            show_centrelines(image, result.vessels, color_by_diameter=True, save_path=save_path)
        elif save_plots and result.vessels:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            if len(image.shape) == 3:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap='gray')
            for vessel in result.vessels:
                ax.plot(vessel.centre[:, 1], vessel.centre[:, 0], 'b-', linewidth=1)
            ax.set_title(f'Centrelines ({len(result.vessels)} segments)')
            ax.axis('off')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved centrelines to: {save_path}")
        
        # Diameter histogram
        if stats['num_measurements'] > 0:
            save_path = os.path.join(output_dir, f"{image_name}_diameter_histogram.png") if save_plots else None
            if show_plots:
                show_diameter_histogram(result, save_path=save_path)
            elif save_plots:
                diameters = analyzer.get_all_diameters(result)
                fig, ax = plt.subplots(1, 1, figsize=(8, 5))
                ax.hist(diameters, bins=50, edgecolor='black', alpha=0.7)
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
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved diameter histogram to: {save_path}")
    
    print("\nDemo complete!")
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Retinal Vessel Segmentation Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic test image
  python demo.py
  
  # Run with your own image
  python demo.py --image path/to/retinal_image.jpg
  
  # Adjust parameters
  python demo.py --image fundus.png --levels 3 4 --threshold 0.15
  
  # Save without displaying
  python demo.py --image fundus.png --no-show
"""
    )
    
    parser.add_argument('--image', '-i', type=str, default=None,
                       help='Path to input image (default: use synthetic test image)')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--levels', '-l', type=int, nargs='+', default=[2, 3],
                       help='Wavelet levels to use (default: 2 3)')
    parser.add_argument('--threshold', '-t', type=float, default=0.20,
                       help='Segmentation threshold percentage (default: 0.20)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots interactively')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots')
    
    args = parser.parse_args()
    
    run_demo(
        image_path=args.image,
        output_dir=args.output,
        wavelet_levels=args.levels,
        threshold=args.threshold,
        show_plots=not args.no_show,
        save_plots=not args.no_save
    )


if __name__ == '__main__':
    main()
