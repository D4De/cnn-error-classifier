from collections import defaultdict
import os
from typing import Any, Dict, Tuple, Union
from args import Args

from coordinates import TensorLayout, map_to_coordinates, numpy_coords_to_python_coord
from domain_classifier import ValueClass, domain_classification, value_classification
from analyzed_tensor import AnalyzedTensor
from spatial_classifier.spatial_classifier import spatial_classification
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

    value_class_count = defaultdict(int)

    # Generate a list of all coordinates where a difference is observed (Sparse matrix)
    sparse_diff_native_coords = list(zip(*np.nonzero(faulty - golden)))
    faulty_native_shape = faulty.shape
    tensor_diff = np.zeros(faulty_native_shape, dtype=np.int8)

    for coord in sparse_diff_native_coords:
        val_class = value_classification(golden[coord[0], coord[1], coord[2], coord[3]], faulty[coord[0], coord[1], coord[2], coord[3]], args.epsilon, args.almost_same)
        tensor_diff[coord[0], coord[1], coord[2], coord[3]] = val_class.value
        value_class_count[val_class] += 1
    
    value_class_count[ValueClass.SAME] = golden.size - sum(value_class_count.values())
        

    # No diff = masked
    if len(sparse_diff_native_coords) == 0:
        log.info(f"{file_path} has no diffs with golden")
        return "masked", None
    sparse_diff = [
        map_to_coordinates(numpy_coords_to_python_coord(coords), args.layout)
        for coords in sparse_diff_native_coords
    ]

    # Pefmorm spatial classifcation
    spatial_class, pattern_params, faulty_channels = spatial_classification(sparse_diff, golden_shape)
    domain_class = domain_classification(value_class_count)


    if args.visualize:
        visualize(
            tensor_diff,
            faulty_channels,
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
        value_classes_counts= value_class_count,
        corrupted_channels_count=len(faulty_channels),
        corrupted_values_count=len(sparse_diff),
        domain_class=domain_class,
        layout=args.layout,
        metadata=metadata
    )
