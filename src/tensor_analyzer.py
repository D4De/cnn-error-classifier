import os
from typing import Any, Dict, Tuple

from coordinates import TensorLayout, map_to_coordinates, numpy_coords_to_python_coord
from domain_classifier import DomainClass
from spatial_classifier import SpatialClass, accumulate_max, spatial_classification
from domain_classifier import domain_classification_vect
import logging as log
import numpy as np

from visualizer import visualize


def analyze_tensor(
    file_path: str,
    golden: np.ndarray,
    layout: TensorLayout,
    epsilon: float,
    almost_same: bool,
    visualize_errors: bool,
    output_dir: str,
    metadata: dict = {},
) -> Tuple[str, Dict[str, Any]]:
    """
    Analyzes a single tensor, in a directory of faulty tensors
    Returns a tuple of two items.
    The first contains the spatial class of the tensor, the second dictionary with various data about the analysis

    file_path: str
    ---
    Path to the faulty tensor

    golden: np.ndarray
    ---
    The golden tensor, that will be compared with the faulty

    layout : TensorLayout
    ---
    The layout in which both the golden and the faulty tensors are stored

    epsilon : float
    ---
    The minimum (absolute value) difference between two values needed to consider
    them different.

    almost_same : bool
    ---
    If true, two different values that have an absolute value difference less than epsilon,
    will be classified as "almost_same". If false, the two values are considered equal. See Args documentation

    visualize_errors : bool
    ---
    If true a visualization of the error locations will be generated and saved as png

    output_dir : str
    ---
    A path to the root of the directory where all outputs will be saved

    metadata : dict
    ---
    A dictionary that contains metadata about the text. The mandatory metadata, needed for processing and classifying the tensor are:
    - bfm: Fault Model
    - igid: Instruction Group Id
    - batch_name: str that indicates the batch name -> contained in info.json

    bfm and igid are derived from the name of the folders that contains faulty tensors. All these folders' names must have the following structure <bfm>_<igid>
    """
    # Initialization
    temp_dom_class_count = np.int64(np.zeros(len(DomainClass)))

    # Opening faulty tensor
    file_name = os.path.basename(file_path).split(".")[0]

    log.debug(f"Opening {file_path}")
    try:
        faulty: np.ndarray = np.load(file_path)
    except:
        log.error(f"Could not read {file_path}")
        return "skipped", {}

    faulty_shape = map_to_coordinates(faulty.shape, layout)
    golden_shape = map_to_coordinates(golden.shape, layout)

    # Check shape correctness
    if faulty_shape != golden_shape:
        log.warn(
            f"Skipping {file_path}. Invalid shape (Faulty has shape: {faulty_shape}, Golden has shape: {golden_shape})"
        )
        return "skipped", {}

    if faulty_shape.N != 1:
        log.warn(f"Skipping {file_path} not supported tensor (Batch dim > 1)")
        return "skipped", {}

    # Execute error domain classification on the whole tensor
    tensor_diff = domain_classification_vect(golden, faulty, epsilon, almost_same)

    # Count occourences of each domain class
    cat, counts = np.unique(tensor_diff, return_counts=True)

    for i in range(len(counts)):
        temp_dom_class_count[cat[i]] = counts[i]

    # Generate a list of all coordinates where a difference is observed (Sparse matrix)
    sparse_diff_native_coords = list(zip(*np.where(tensor_diff > 0)))
    (raveld_coords,) = np.where(np.ravel(tensor_diff) > 1)

    # No diff = masked
    if len(sparse_diff_native_coords) == 0:
        log.info(f"{file_path} has no diffs with golden")
        return "masked", {}
    sparse_diff = [
        map_to_coordinates(numpy_coords_to_python_coord(coords), layout)
        for coords in sparse_diff_native_coords
    ]

    # Pefmorm spatial classifcation
    spatial_class, pattern_params = spatial_classification(sparse_diff, golden_shape)

    # Get faults for each channel
    # faulty_channels = {coord.C for coord in sparse_diff}
    if layout == TensorLayout.NCHW:
        channel_sums = np.sum(tensor_diff, axis=(0, 2, 3))
    elif layout == TensorLayout.NHWC:
        channel_sums = np.sum(tensor_diff, axis=(0, 1, 2))

    (faulty_channels,) = np.where(channel_sums != 0)

    if visualize_errors:
        visualize(
            tensor_diff,
            faulty_channels.tolist(),
            layout,
            spatial_class.output_path(
                output_dir, f'{metadata["batch_name"]}_{file_name}'
            ),
            save=True,
            show=False,
            suptitile=f'{metadata.get("batch_name") or ""} {metadata.get("igid") or ""} {metadata.get("bfm") or ""} {golden_shape.C}x{golden_shape.H}x{golden_shape.W}',
            invalidate=True,
        )

    raveled_offsets = (raveld_coords - raveld_coords[0]).tolist()
    # Per tensor report generator
    return spatial_class.display_name(), {
        "class": spatial_class.display_name(),
        "class_params": pattern_params,
        "domain_classes": {
            clz.display_name(): temp_dom_class_count[i].item()
            for i, clz in enumerate(DomainClass)
        },
        "corrupted_values": sum(temp_dom_class_count[2:]).item(),
        "corrupted_values_pct": sum(temp_dom_class_count[2:]).item()
        / golden.size
        * 100,
        "affected_channels": faulty_channels.tolist(),
        "faulty_channels_count": len(faulty_channels),
        "raveled_start": raveld_coords[0].item(),
        "raveled_offsets": sorted(raveled_offsets),
        "error_pattern": pattern_params["error_pattern"],
        "block_align": pattern_params["align"]
        if spatial_class == SpatialClass.CHANNEL_ALIGNED_SAME_BLOCK
        else None,
    }
