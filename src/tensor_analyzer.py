import os
from typing import Any, Dict, Tuple, Union
from args import Args

from coordinates import TensorLayout, map_to_coordinates, numpy_coords_to_python_coord
from domain_classifier import DomainClass
from analyzed_tensor import AnalyzedTensor
from spatial_classifier import spatial_classification
from domain_classifier import domain_classification_vect
import logging as log
import numpy as np

from visualizer import visualize


def analyze_tensor(
    file_path: str,
    golden: np.ndarray,
    args: Args,
    metadata: dict = {},
) -> Tuple[str, Union[AnalyzedTensor, None]]:
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
        return "skipped", None

    faulty_shape = map_to_coordinates(faulty.shape, args.layout)
    golden_shape = map_to_coordinates(golden.shape, args.layout)

    # Check shape correctness
    if faulty_shape != golden_shape:
        log.warn(
            f"Skipping {file_path}. Invalid shape (Faulty has shape: {faulty_shape}, Golden has shape: {golden_shape})"
        )
        return "skipped", None

    if faulty_shape.N != 1:
        log.warn(f"Skipping {file_path} not supported tensor (Batch dim > 1)")
        return "skipped", None

    # Execute error domain classification on the whole tensor
    tensor_diff = domain_classification_vect(golden, faulty, args.epsilon, args.almost_same)

    # Count occourences of each domain class
    cat, counts = np.unique(tensor_diff, return_counts=True)

    for i in range(len(counts)):
        temp_dom_class_count[cat[i]] = counts[i]

    # Generate a list of all coordinates where a difference is observed (Sparse matrix)
    sparse_diff_native_coords = list(zip(*np.where(tensor_diff > 0)))

    # No diff = masked
    if len(sparse_diff_native_coords) == 0:
        log.info(f"{file_path} has no diffs with golden")
        return "masked", None
    sparse_diff = [
        map_to_coordinates(numpy_coords_to_python_coord(coords), args.layout)
        for coords in sparse_diff_native_coords
    ]

    # Pefmorm spatial classifcation
    spatial_class, pattern_params = spatial_classification(sparse_diff, golden_shape)

    # Get faults for each channel
    # faulty_channels = {coord.C for coord in sparse_diff}
    if args.layout == TensorLayout.NCHW:
        channel_sums = np.sum(tensor_diff, axis=(0, 2, 3))
    elif args.layout == TensorLayout.NHWC:
        channel_sums = np.sum(tensor_diff, axis=(0, 1, 2))

    (faulty_channels,) = np.where(channel_sums != 0)

    if args.visualize:
        visualize(
            tensor_diff,
            faulty_channels.tolist(),
            args.layout,
            spatial_class.output_path(
                args.visualize_path, f'{metadata["batch_name"]}_{file_name}'
            ),
            save=True,
            show=False,
            suptitile=f'{metadata.get("batch_name") or ""} {metadata.get("igid") or ""} {metadata.get("bfm") or ""} {golden_shape.C}x{golden_shape.H}x{golden_shape.W}',
            invalidate=True,
        )

    # Per tensor report generator
    return spatial_class.display_name(), AnalyzedTensor(
        batch=metadata["batch_name"],
        sub_batch=metadata["sub_batch_name"],
        file_name=os.path.basename(file_path),
        file_path=file_path,
        shape=faulty_shape,
        spatial_class=spatial_class,
        spatial_class_params=pattern_params,
        spatial_pattern=pattern_params.get("error_pattern"),
        domain_classes_counts= {
            clz: temp_dom_class_count[i].item()
            for i, clz in enumerate(DomainClass)
        },
        corrupted_channels_count=len(set(x.C for x in sparse_diff)),
        corrupted_values_count=len(sparse_diff),
        layout=args.layout,
        metadata=metadata
    )
