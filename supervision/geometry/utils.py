import numpy as np
from supervision.geometry.core import Point
from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes


def get_polygon_center(polygon: np.ndarray) -> Point:
    """
    Calculate the center of a polygon. The center is calculated as the center
    of the solid figure formed by the points of the polygon

    Parameters:
        polygon (np.ndarray): A 2-dimensional numpy ndarray representing the
            vertices of the polygon.

    Returns:
        Point: The center of the polygon, represented as a
            Point object with x and y attributes.

    Raises:
        ValueError: If the polygon has no vertices.

    Examples:
        ```python
        import numpy as np
        import supervision as sv

        polygon = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        sv.get_polygon_center(polygon=polygon)
        # Point(x=1, y=1)
        ```
    """

    # This is one of the 3 candidate algorithms considered for centroid calculation.
    # For a more detailed discussion, see PR #1084 and commit eb33176

    if len(polygon) == 0:
        raise ValueError("Polygon must have at least one vertex.")

    shift_polygon = np.roll(polygon, -1, axis=0)
    signed_areas = np.cross(polygon, shift_polygon) / 2
    if signed_areas.sum() == 0:
        center = np.mean(polygon, axis=0).round()
        return Point(x=center[0], y=center[1])
    centroids = (polygon + shift_polygon) / 3.0
    center = np.average(centroids, axis=0, weights=signed_areas).round()

    return Point(x=center[0], y=center[1])


def extract_centroids_from_segmentations(polygons, img_height, img_width, bounding_boxes):
    """Calculate centroids from COCO polygon annotations.

    Args:
        polygons: List of COCO polygon annotations representing object shapes.
        img_height: Height of the image to create the mask.
        img_width: Width of the image to create the mask.
        bounding_boxes: List of bounding boxes for each polygon.

    Returns:
        A PyTorch tensor containing the calculated centroids of the objects.
    """
    masks_list = []
    estimated_centroids = []
    
    for poly in polygons:
        # Create a blank mask image
        mask_image = Image.new('L', (img_width, img_height), 0)
        ImageDraw.Draw(mask_image).polygon(poly[0], outline=1, fill=1)
        mask_array = np.array(mask_image)
        masks_list.append(mask_array)

        # Calculate initial centroid
        y_coords, x_coords = np.where(mask_array == 1)
        centroid_estimate = [np.mean(x_coords), np.mean(y_coords)]
        estimated_centroids.append(centroid_estimate)

    refined_masks = []
    for idx, mask in enumerate(masks_list):
        height_diff = abs(bounding_boxes[idx][0] - bounding_boxes[idx][2]).item()
        width_diff = abs(bounding_boxes[idx][1] - bounding_boxes[idx][3]).item()
        min_threshold = int(min(height_diff, width_diff) / 8)

        # Erode and then dilate the mask
        refined_mask = binary_dilation(binary_erosion(mask, iterations=min_threshold), iterations=min_threshold)
        refined_masks.append(refined_mask.astype(int))

    centroid_positions = []
    for idx, refined_mask in enumerate(refined_masks):
        if np.sum(refined_mask) == 0:
            centroid = estimated_centroids[idx]
        else:
            y_coords, x_coords = np.where(refined_mask == 1)
            if x_coords.size == 0 or y_coords.size == 0:
                centroid = estimated_centroids[idx]
            else:
                centroid = [np.mean(x_coords), np.mean(y_coords)]
        centroid_positions.append(centroid)

    return torch.tensor(centroid_positions, dtype=torch.float)


