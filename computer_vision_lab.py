"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

COMPUTER VISION LAB
Production-ready computer vision algorithms implemented from scratch.
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ComputerVisionConfig:
    """Configuration for computer vision operations."""
    image_size: Tuple[int, int] = (256, 256)
    channels: int = 3
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    pooling_size: int = 2
    gaussian_sigma: float = 1.0
    canny_low_threshold: float = 50
    canny_high_threshold: float = 150
    sift_octaves: int = 4
    sift_scales_per_octave: int = 3
    harris_k: float = 0.04
    harris_threshold: float = 0.01
    nms_threshold: float = 0.5
    iou_threshold: float = 0.5

class ComputerVisionLab:
    """Production-ready computer vision algorithms implemented from scratch."""

    def __init__(self, config: ComputerVisionConfig = None):
        self.config = config or ComputerVisionConfig()
        self.kernels = self._init_kernels()

    def _init_kernels(self) -> Dict[str, np.ndarray]:
        """Initialize commonly used convolution kernels."""
        kernels = {}

        # Sobel operators
        kernels['sobel_x'] = np.array([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=np.float64)

        kernels['sobel_y'] = np.array([[1, 2, 1],
                                       [0, 0, 0],
                                       [-1, -2, -1]], dtype=np.float64)

        # Prewitt operators
        kernels['prewitt_x'] = np.array([[-1, 0, 1],
                                         [-1, 0, 1],
                                         [-1, 0, 1]], dtype=np.float64)

        kernels['prewitt_y'] = np.array([[1, 1, 1],
                                         [0, 0, 0],
                                         [-1, -1, -1]], dtype=np.float64)

        # Laplacian
        kernels['laplacian'] = np.array([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=np.float64)

        # Gaussian blur (3x3)
        kernels['gaussian_3x3'] = np.array([[1, 2, 1],
                                           [2, 4, 2],
                                           [1, 2, 1]], dtype=np.float64) / 16

        # Sharpening
        kernels['sharpen'] = np.array([[0, -1, 0],
                                       [-1, 5, -1],
                                       [0, -1, 0]], dtype=np.float64)

        return kernels

    def convolution2d(self, image: np.ndarray, kernel: np.ndarray,
                     stride: int = 1, padding: int = 0) -> np.ndarray:
        """
        Perform 2D convolution operation from scratch.

        Args:
            image: Input image (H, W) or (H, W, C)
            kernel: Convolution kernel
            stride: Stride for convolution
            padding: Padding size

        Returns:
            Convolved image
        """
        if len(image.shape) == 3:
            # Apply convolution to each channel
            result = np.zeros_like(image)
            for c in range(image.shape[2]):
                result[:, :, c] = self.convolution2d(image[:, :, c], kernel, stride, padding)
            return result

        # Add padding
        if padding > 0:
            image = np.pad(image, padding, mode='constant')

        h, w = image.shape
        kh, kw = kernel.shape
        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1

        output = np.zeros((out_h, out_w), dtype=np.float64)

        for i in range(out_h):
            for j in range(out_w):
                y = i * stride
                x = j * stride
                output[i, j] = np.sum(image[y:y+kh, x:x+kw] * kernel)

        return output

    def max_pooling2d(self, image: np.ndarray, pool_size: int = 2,
                     stride: Optional[int] = None) -> np.ndarray:
        """
        Perform max pooling operation.

        Args:
            image: Input image
            pool_size: Size of pooling window
            stride: Stride (defaults to pool_size)

        Returns:
            Pooled image
        """
        if stride is None:
            stride = pool_size

        if len(image.shape) == 3:
            h, w, c = image.shape
            out_h = (h - pool_size) // stride + 1
            out_w = (w - pool_size) // stride + 1
            output = np.zeros((out_h, out_w, c), dtype=np.float64)

            for ch in range(c):
                output[:, :, ch] = self.max_pooling2d(image[:, :, ch], pool_size, stride)
            return output

        h, w = image.shape
        out_h = (h - pool_size) // stride + 1
        out_w = (w - pool_size) // stride + 1
        output = np.zeros((out_h, out_w), dtype=np.float64)

        for i in range(out_h):
            for j in range(out_w):
                y = i * stride
                x = j * stride
                output[i, j] = np.max(image[y:y+pool_size, x:x+pool_size])

        return output

    def average_pooling2d(self, image: np.ndarray, pool_size: int = 2,
                         stride: Optional[int] = None) -> np.ndarray:
        """Perform average pooling operation."""
        if stride is None:
            stride = pool_size

        if len(image.shape) == 3:
            h, w, c = image.shape
            out_h = (h - pool_size) // stride + 1
            out_w = (w - pool_size) // stride + 1
            output = np.zeros((out_h, out_w, c), dtype=np.float64)

            for ch in range(c):
                output[:, :, ch] = self.average_pooling2d(image[:, :, ch], pool_size, stride)
            return output

        h, w = image.shape
        out_h = (h - pool_size) // stride + 1
        out_w = (w - pool_size) // stride + 1
        output = np.zeros((out_h, out_w), dtype=np.float64)

        for i in range(out_h):
            for j in range(out_w):
                y = i * stride
                x = j * stride
                output[i, j] = np.mean(image[y:y+pool_size, x:x+pool_size])

        return output

    def gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Generate Gaussian kernel for blurring."""
        kernel = np.zeros((size, size), dtype=np.float64)
        center = size // 2

        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

        return kernel / np.sum(kernel)

    def gaussian_blur(self, image: np.ndarray, kernel_size: int = 5,
                     sigma: Optional[float] = None) -> np.ndarray:
        """Apply Gaussian blur to image."""
        if sigma is None:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        kernel = self.gaussian_kernel(kernel_size, sigma)
        return self.convolution2d(image, kernel, padding=kernel_size//2)

    def sobel_edge_detection(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform Sobel edge detection.

        Args:
            image: Input grayscale image

        Returns:
            Gradient magnitude, gradient direction, edge map
        """
        # Apply Sobel operators
        grad_x = self.convolution2d(image, self.kernels['sobel_x'], padding=1)
        grad_y = self.convolution2d(image, self.kernels['sobel_y'], padding=1)

        # Calculate magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Threshold to get edge map
        threshold = np.mean(magnitude) + np.std(magnitude)
        edges = (magnitude > threshold).astype(np.float64)

        return magnitude, direction, edges

    def canny_edge_detection(self, image: np.ndarray,
                           low_threshold: Optional[float] = None,
                           high_threshold: Optional[float] = None) -> np.ndarray:
        """
        Implement Canny edge detection algorithm.

        Args:
            image: Input grayscale image
            low_threshold: Low threshold for edge linking
            high_threshold: High threshold for edge detection

        Returns:
            Binary edge map
        """
        if low_threshold is None:
            low_threshold = self.config.canny_low_threshold
        if high_threshold is None:
            high_threshold = self.config.canny_high_threshold

        # Step 1: Gaussian blur
        blurred = self.gaussian_blur(image, kernel_size=5, sigma=1.4)

        # Step 2: Gradient calculation
        grad_x = self.convolution2d(blurred, self.kernels['sobel_x'], padding=1)
        grad_y = self.convolution2d(blurred, self.kernels['sobel_y'], padding=1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)

        # Step 3: Non-maximum suppression
        nms = self._non_maximum_suppression(magnitude, direction)

        # Step 4: Double thresholding
        strong_edges = nms > high_threshold
        weak_edges = (nms >= low_threshold) & (nms <= high_threshold)

        # Step 5: Edge tracking by hysteresis
        edges = self._edge_tracking_hysteresis(strong_edges, weak_edges)

        return edges.astype(np.float64)

    def _non_maximum_suppression(self, magnitude: np.ndarray,
                                direction: np.ndarray) -> np.ndarray:
        """Apply non-maximum suppression to thin edges."""
        h, w = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        angle = direction * 180.0 / np.pi
        angle[angle < 0] += 180

        for i in range(1, h-1):
            for j in range(1, w-1):
                q = 255
                r = 255

                # Angle quantization
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]

                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    suppressed[i, j] = magnitude[i, j]

        return suppressed

    def _edge_tracking_hysteresis(self, strong: np.ndarray,
                                 weak: np.ndarray) -> np.ndarray:
        """Track edges using hysteresis."""
        edges = strong.copy()
        h, w = edges.shape

        # Find weak edges connected to strong edges
        for i in range(1, h-1):
            for j in range(1, w-1):
                if weak[i, j]:
                    # Check 8-connectivity
                    if np.any(edges[i-1:i+2, j-1:j+2]):
                        edges[i, j] = 1

        return edges

    def harris_corner_detection(self, image: np.ndarray,
                              k: Optional[float] = None,
                              threshold: Optional[float] = None) -> np.ndarray:
        """
        Implement Harris corner detection.

        Args:
            image: Input grayscale image
            k: Harris detector parameter
            threshold: Threshold for corner detection

        Returns:
            Corner response map
        """
        if k is None:
            k = self.config.harris_k
        if threshold is None:
            threshold = self.config.harris_threshold

        # Calculate gradients
        Ix = self.convolution2d(image, self.kernels['sobel_x'], padding=1)
        Iy = self.convolution2d(image, self.kernels['sobel_y'], padding=1)

        # Calculate products of derivatives
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # Apply Gaussian weighting
        kernel_size = 3
        Ixx = self.gaussian_blur(Ixx, kernel_size)
        Iyy = self.gaussian_blur(Iyy, kernel_size)
        Ixy = self.gaussian_blur(Ixy, kernel_size)

        # Calculate Harris response
        det = Ixx * Iyy - Ixy**2
        trace = Ixx + Iyy
        response = det - k * trace**2

        # Threshold and non-maximum suppression
        corners = np.zeros_like(response)
        corners[response > threshold * response.max()] = 1

        return corners

    def sift_like_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract SIFT-like features (simplified version).

        Args:
            image: Input grayscale image

        Returns:
            Dictionary with keypoints and descriptors
        """
        # Build scale space (Gaussian pyramid)
        octaves = []
        current = image.copy()

        for octave in range(self.config.sift_octaves):
            scales = []
            sigma = 1.6

            for scale in range(self.config.sift_scales_per_octave):
                blurred = self.gaussian_blur(current, kernel_size=5, sigma=sigma)
                scales.append(blurred)
                sigma *= 1.4

            octaves.append(scales)
            current = current[::2, ::2]  # Downsample

        # Find keypoints (simplified - using Harris corners at multiple scales)
        keypoints = []

        for octave_idx, octave in enumerate(octaves):
            for scale_idx, scale_image in enumerate(octave):
                corners = self.harris_corner_detection(scale_image)
                coords = np.where(corners > 0)

                for y, x in zip(coords[0], coords[1]):
                    # Scale coordinates back to original image
                    scale_factor = 2 ** octave_idx
                    keypoints.append({
                        'x': x * scale_factor,
                        'y': y * scale_factor,
                        'scale': scale_idx,
                        'octave': octave_idx,
                        'response': corners[y, x]
                    })

        # Extract descriptors (simplified - using gradient histograms)
        descriptors = []

        for kp in keypoints[:100]:  # Limit for demo
            x, y = int(kp['x']), int(kp['y'])
            patch_size = 16

            # Extract patch around keypoint
            y_min = max(0, y - patch_size // 2)
            y_max = min(image.shape[0], y + patch_size // 2)
            x_min = max(0, x - patch_size // 2)
            x_max = min(image.shape[1], x + patch_size // 2)

            if y_max - y_min < 4 or x_max - x_min < 4:
                continue

            patch = image[y_min:y_max, x_min:x_max]

            # Calculate gradient histogram (4x4 grid, 8 orientations)
            descriptor = self._compute_gradient_histogram(patch, 4, 8)
            descriptors.append(descriptor)

        return {
            'keypoints': keypoints[:len(descriptors)],
            'descriptors': np.array(descriptors)
        }

    def _compute_gradient_histogram(self, patch: np.ndarray,
                                  grid_size: int, n_orientations: int) -> np.ndarray:
        """Compute gradient histogram descriptor for a patch."""
        h, w = patch.shape
        cell_h = h // grid_size
        cell_w = w // grid_size

        # Calculate gradients
        grad_x = self.convolution2d(patch, self.kernels['sobel_x'], padding=1)
        grad_y = self.convolution2d(patch, self.kernels['sobel_y'], padding=1)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)

        # Quantize orientations
        orientation_bins = (orientation + np.pi) * n_orientations / (2 * np.pi)
        orientation_bins = np.floor(orientation_bins).astype(int) % n_orientations

        # Build histogram for each cell
        descriptor = []

        for i in range(grid_size):
            for j in range(grid_size):
                cell_hist = np.zeros(n_orientations)

                y_start = i * cell_h
                y_end = min((i + 1) * cell_h, h)
                x_start = j * cell_w
                x_end = min((j + 1) * cell_w, w)

                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation_bins[y_start:y_end, x_start:x_end]

                for ori in range(n_orientations):
                    cell_hist[ori] = np.sum(cell_mag[cell_ori == ori])

                descriptor.extend(cell_hist)

        # Normalize descriptor
        descriptor = np.array(descriptor)
        norm = np.linalg.norm(descriptor)
        if norm > 0:
            descriptor /= norm

        return descriptor

    def image_augmentation(self, image: np.ndarray,
                         augmentation_type: str = 'all') -> np.ndarray:
        """
        Apply various image augmentation techniques.

        Args:
            image: Input image
            augmentation_type: Type of augmentation

        Returns:
            Augmented image
        """
        augmented = image.copy()

        if augmentation_type in ['flip_h', 'all']:
            # Horizontal flip
            if np.random.rand() > 0.5:
                augmented = np.fliplr(augmented)

        if augmentation_type in ['flip_v', 'all']:
            # Vertical flip
            if np.random.rand() > 0.5:
                augmented = np.flipud(augmented)

        if augmentation_type in ['rotate', 'all']:
            # Random rotation (90 degree increments)
            k = np.random.randint(0, 4)
            augmented = np.rot90(augmented, k)

        if augmentation_type in ['noise', 'all']:
            # Add Gaussian noise
            noise = np.random.randn(*augmented.shape) * 0.05
            augmented = np.clip(augmented + noise, 0, 1)

        if augmentation_type in ['brightness', 'all']:
            # Adjust brightness
            factor = np.random.uniform(0.7, 1.3)
            augmented = np.clip(augmented * factor, 0, 1)

        if augmentation_type in ['contrast', 'all']:
            # Adjust contrast
            factor = np.random.uniform(0.5, 1.5)
            mean = np.mean(augmented)
            augmented = np.clip((augmented - mean) * factor + mean, 0, 1)

        return augmented

    def intersection_over_union(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) for bounding boxes.

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU score
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray,
                          threshold: Optional[float] = None) -> List[int]:
        """
        Apply Non-Maximum Suppression to bounding boxes.

        Args:
            boxes: Array of boxes [N, 4] (x1, y1, x2, y2)
            scores: Confidence scores [N]
            threshold: IoU threshold

        Returns:
            Indices of kept boxes
        """
        if threshold is None:
            threshold = self.config.nms_threshold

        if len(boxes) == 0:
            return []

        # Sort by scores
        indices = np.argsort(scores)[::-1]
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # Calculate IoU with remaining boxes
            ious = np.array([self.intersection_over_union(boxes[current], boxes[idx])
                           for idx in indices[1:]])

            # Keep boxes with IoU below threshold
            indices = indices[1:][ious < threshold]

        return keep

    def mean_average_precision(self, predictions: List[Dict],
                             ground_truth: List[Dict],
                             iou_threshold: Optional[float] = None) -> float:
        """
        Calculate mean Average Precision (mAP) for object detection.

        Args:
            predictions: List of predicted boxes with scores and classes
            ground_truth: List of ground truth boxes with classes
            iou_threshold: IoU threshold for matching

        Returns:
            mAP score
        """
        if iou_threshold is None:
            iou_threshold = self.config.iou_threshold

        # Group by class
        classes = set()
        for pred in predictions:
            classes.add(pred['class'])
        for gt in ground_truth:
            classes.add(gt['class'])

        average_precisions = []

        for cls in classes:
            # Filter predictions and ground truth for this class
            cls_preds = [p for p in predictions if p['class'] == cls]
            cls_gt = [g for g in ground_truth if g['class'] == cls]

            if len(cls_gt) == 0:
                continue

            # Sort predictions by score
            cls_preds.sort(key=lambda x: x['score'], reverse=True)

            # Match predictions to ground truth
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            gt_matched = set()

            for i, pred in enumerate(cls_preds):
                best_iou = 0
                best_gt_idx = -1

                for j, gt in enumerate(cls_gt):
                    if j in gt_matched:
                        continue

                    iou = self.intersection_over_union(pred['box'], gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j

                if best_iou >= iou_threshold:
                    tp[i] = 1
                    gt_matched.add(best_gt_idx)
                else:
                    fp[i] = 1

            # Calculate precision and recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls = tp_cumsum / len(cls_gt)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

            # Calculate AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11

            average_precisions.append(ap)

        return np.mean(average_precisions) if average_precisions else 0.0

    def semantic_segmentation_metrics(self, pred_mask: np.ndarray,
                                    true_mask: np.ndarray,
                                    n_classes: int) -> Dict[str, float]:
        """
        Calculate metrics for semantic segmentation.

        Args:
            pred_mask: Predicted segmentation mask
            true_mask: Ground truth mask
            n_classes: Number of classes

        Returns:
            Dictionary with IoU, Dice coefficient, and pixel accuracy
        """
        metrics = {}

        # Pixel accuracy
        correct_pixels = np.sum(pred_mask == true_mask)
        total_pixels = pred_mask.size
        metrics['pixel_accuracy'] = correct_pixels / total_pixels

        # Per-class IoU and Dice
        ious = []
        dice_scores = []

        for cls in range(n_classes):
            pred_cls = pred_mask == cls
            true_cls = true_mask == cls

            intersection = np.sum(pred_cls & true_cls)
            union = np.sum(pred_cls | true_cls)

            if union > 0:
                iou = intersection / union
                ious.append(iou)

                dice = 2 * intersection / (np.sum(pred_cls) + np.sum(true_cls) + 1e-10)
                dice_scores.append(dice)

        metrics['mean_iou'] = np.mean(ious) if ious else 0.0
        metrics['mean_dice'] = np.mean(dice_scores) if dice_scores else 0.0

        return metrics

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization for contrast enhancement."""
        # Convert to uint8 for histogram calculation
        img_uint8 = (image * 255).astype(np.uint8)

        # Calculate histogram
        hist, _ = np.histogram(img_uint8.flatten(), bins=256, range=(0, 256))

        # Calculate CDF
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf[-1]

        # Apply equalization
        equalized = np.interp(img_uint8.flatten(), range(256), cdf_normalized * 255)
        equalized = equalized.reshape(image.shape)

        return equalized / 255.0

    def template_matching(self, image: np.ndarray, template: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Perform template matching using normalized cross-correlation.

        Args:
            image: Input image
            template: Template to find

        Returns:
            Correlation map and best match location
        """
        h_img, w_img = image.shape[:2]
        h_tmp, w_tmp = template.shape[:2]

        # Ensure template is smaller than image
        if h_tmp > h_img or w_tmp > w_img:
            raise ValueError("Template must be smaller than image")

        correlation_map = np.zeros((h_img - h_tmp + 1, w_img - w_tmp + 1))

        # Normalize template
        template_norm = template - np.mean(template)
        template_std = np.std(template)

        for i in range(correlation_map.shape[0]):
            for j in range(correlation_map.shape[1]):
                # Extract window
                window = image[i:i+h_tmp, j:j+w_tmp]

                # Normalize window
                window_norm = window - np.mean(window)
                window_std = np.std(window)

                # Calculate normalized cross-correlation
                if template_std > 0 and window_std > 0:
                    correlation = np.sum(template_norm * window_norm) / (h_tmp * w_tmp * template_std * window_std)
                    correlation_map[i, j] = correlation

        # Find best match
        best_match = np.unravel_index(np.argmax(correlation_map), correlation_map.shape)

        return correlation_map, best_match

def run_demo():
    """Demonstrate the computer vision lab capabilities."""
    print("=" * 80)
    print("COMPUTER VISION LAB - Production Demo")
    print("Copyright (c) 2025 Corporation of Light")
    print("=" * 80)

    # Initialize lab
    cv_lab = ComputerVisionLab()

    # Create synthetic test image
    np.random.seed(42)
    height, width = 128, 128

    # Create gradient image with some features
    image = np.zeros((height, width), dtype=np.float64)

    # Add gradient
    for i in range(height):
        for j in range(width):
            image[i, j] = (i + j) / (height + width)

    # Add some rectangles and circles
    image[20:40, 20:60] = 0.8  # Rectangle
    image[60:90, 70:100] = 0.9  # Another rectangle

    # Add circle (approximate)
    center = (64, 64)
    radius = 20
    for i in range(height):
        for j in range(width):
            if np.sqrt((i - center[0])**2 + (j - center[1])**2) < radius:
                image[i, j] = 0.7

    print("\n1. CONVOLUTION OPERATIONS")
    print("-" * 40)

    # Apply different kernels
    gaussian_result = cv_lab.convolution2d(image, cv_lab.kernels['gaussian_3x3'], padding=1)
    sharpened = cv_lab.convolution2d(image, cv_lab.kernels['sharpen'], padding=1)
    laplacian = cv_lab.convolution2d(image, cv_lab.kernels['laplacian'], padding=1)

    print(f"   Original image shape: {image.shape}")
    print(f"   Gaussian blur mean: {np.mean(gaussian_result):.4f}")
    print(f"   Sharpened std: {np.std(sharpened):.4f}")
    print(f"   Laplacian range: [{np.min(laplacian):.4f}, {np.max(laplacian):.4f}]")

    print("\n2. EDGE DETECTION ALGORITHMS")
    print("-" * 40)

    # Sobel edge detection
    mag, direction, edges = cv_lab.sobel_edge_detection(image)
    print(f"   Sobel magnitude range: [{np.min(mag):.4f}, {np.max(mag):.4f}]")
    print(f"   Edge pixels detected: {np.sum(edges):.0f}")

    # Canny edge detection
    canny_edges = cv_lab.canny_edge_detection(image)
    print(f"   Canny edge pixels: {np.sum(canny_edges):.0f}")

    print("\n3. POOLING OPERATIONS")
    print("-" * 40)

    # Create test image for pooling
    test_img = np.random.randn(64, 64) * 0.5 + 0.5

    max_pooled = cv_lab.max_pooling2d(test_img, pool_size=2)
    avg_pooled = cv_lab.average_pooling2d(test_img, pool_size=2)

    print(f"   Original shape: {test_img.shape}")
    print(f"   Max pooled shape: {max_pooled.shape}")
    print(f"   Avg pooled shape: {avg_pooled.shape}")
    print(f"   Max pooling preserves: {np.max(max_pooled):.4f} (max value)")

    print("\n4. CORNER DETECTION")
    print("-" * 40)

    corners = cv_lab.harris_corner_detection(image)
    corner_count = np.sum(corners > 0)
    print(f"   Harris corners detected: {corner_count}")

    print("\n5. FEATURE EXTRACTION (SIFT-like)")
    print("-" * 40)

    features = cv_lab.sift_like_features(image)
    print(f"   Keypoints found: {len(features['keypoints'])}")
    if len(features['descriptors']) > 0:
        print(f"   Descriptor size: {features['descriptors'].shape[1]}")
        print(f"   Descriptor norm (first): {np.linalg.norm(features['descriptors'][0]):.4f}")

    print("\n6. IMAGE AUGMENTATION")
    print("-" * 40)

    augmentation_types = ['flip_h', 'rotate', 'noise', 'brightness']
    for aug_type in augmentation_types:
        augmented = cv_lab.image_augmentation(image, aug_type)
        diff = np.mean(np.abs(augmented - image))
        print(f"   {aug_type:12s}: Mean difference = {diff:.4f}")

    print("\n7. OBJECT DETECTION METRICS")
    print("-" * 40)

    # Test IoU calculation
    box1 = np.array([10, 10, 50, 50])
    box2 = np.array([30, 30, 70, 70])
    iou = cv_lab.intersection_over_union(box1, box2)
    print(f"   IoU between test boxes: {iou:.4f}")

    # Test NMS
    boxes = np.array([[10, 10, 30, 30],
                     [15, 15, 35, 35],
                     [50, 50, 70, 70],
                     [55, 55, 75, 75]])
    scores = np.array([0.9, 0.8, 0.95, 0.7])
    keep_indices = cv_lab.non_max_suppression(boxes, scores, threshold=0.5)
    print(f"   NMS kept {len(keep_indices)} out of {len(boxes)} boxes")

    # Test mAP
    predictions = [
        {'box': np.array([10, 10, 30, 30]), 'score': 0.9, 'class': 0},
        {'box': np.array([50, 50, 70, 70]), 'score': 0.8, 'class': 1}
    ]
    ground_truth = [
        {'box': np.array([12, 12, 32, 32]), 'class': 0},
        {'box': np.array([48, 48, 68, 68]), 'class': 1}
    ]
    mAP = cv_lab.mean_average_precision(predictions, ground_truth)
    print(f"   Mean Average Precision: {mAP:.4f}")

    print("\n8. SEGMENTATION METRICS")
    print("-" * 40)

    # Create synthetic segmentation masks
    pred_mask = np.random.randint(0, 3, size=(64, 64))
    true_mask = np.random.randint(0, 3, size=(64, 64))

    # Make them somewhat similar
    pred_mask[:32, :32] = true_mask[:32, :32]

    seg_metrics = cv_lab.semantic_segmentation_metrics(pred_mask, true_mask, n_classes=3)
    print(f"   Pixel accuracy: {seg_metrics['pixel_accuracy']:.4f}")
    print(f"   Mean IoU: {seg_metrics['mean_iou']:.4f}")
    print(f"   Mean Dice: {seg_metrics['mean_dice']:.4f}")

    print("\n9. HISTOGRAM EQUALIZATION")
    print("-" * 40)

    # Create low contrast image
    low_contrast = image * 0.3 + 0.3
    equalized = cv_lab.histogram_equalization(low_contrast)

    print(f"   Original range: [{np.min(low_contrast):.4f}, {np.max(low_contrast):.4f}]")
    print(f"   Equalized range: [{np.min(equalized):.4f}, {np.max(equalized):.4f}]")
    print(f"   Contrast improvement: {np.std(equalized) / np.std(low_contrast):.2f}x")

    print("\n10. TEMPLATE MATCHING")
    print("-" * 40)

    # Create template from part of image
    template = image[60:80, 70:90]
    correlation_map, best_match = cv_lab.template_matching(image, template)

    print(f"   Template size: {template.shape}")
    print(f"   Best match location: {best_match}")
    print(f"   Max correlation: {np.max(correlation_map):.4f}")

    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    run_demo()