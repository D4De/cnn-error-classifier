from collections import defaultdict
import os
from typing import Optional, Tuple
from args import Args

from coordinates import map_to_coordinates, numpy_coords_to_python_coord
from domain_classifier import ValueClass, domain_classification, value_classification
from analyzed_tensor import AnalyzedTensor
from spatial_classifier.spatial_classifier import spatial_classification
import logging as log
import numpy as np

from visualizer import visualize


def analyze_tensor(
    file_path: str,
    golden: np.ndarray,
    golden_range_min: float,
    golden_range_max: float,
    args: Args,
    metadata: dict = {},
) -> Tuple[str, Optional[AnalyzedTensor]]:
    """
    Analyzes a single tensor, in a directory of faulty tensors

    Parameters
    ---
    file_path : str
        A relative or absolute path to the corrupted tensor to analyze
    golden : ndarray
        A numpy array containing the golden tensor, to compare with the corrupted one
    golden_range_min : float
        The lowest value in the golden tensor
    golden_range_max : float
        The highest value in the golden tensor     
    args : Args
        Object containing all the user preferences supplied by the command line arguments
    metadata : dict
        Extra data on how the experiments were conducted, contained in a key-value dict.
        "batch_name" and "sub_batch_name" keys are required, the rest are not needed in order
        to make the classification work.

        In particular the values associated to "batch_name" and "sub_batch_name" 
        will appear on the visulizer plots, and are used in the output database 


    Returns
    ---
    Depending on the outcome of the analysis:
        * If the corrupted tensor npy file fails to open or has a different shape than the golden:
            * ("skipped", None) will be returned
        * if the corrupted tensor is equal or almost equal (all differences under args.epsilon) to the golden one
            * ("masked", None)
        * otherwise, if the analysis succeeds:
            * (<sp_class_display_name>, AnalyzedTensor object) where:
                * <sp_class_display_name> is the display name of the spatial class to which the corrupted tensor belongs in relation to the golden one
                * AnalyzedTensor object: An object containing all the details of the analysis

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
    if args.almost_same:
        sparse_diff_native_coords = list(zip(*np.nonzero(faulty - golden)))
    else:
        sparse_diff_native_coords = list(zip(*np.where(np.abs(faulty - golden) >= args.epsilon)))
    faulty_native_shape = faulty.shape
    tensor_diff = np.zeros(faulty_native_shape, dtype=np.int8)

    for coord in sparse_diff_native_coords:
        val_class = value_classification(golden[coord[0], coord[1], coord[2], coord[3]], faulty[coord[0], coord[1], coord[2], coord[3]], golden_range_min, golden_range_max,  args.epsilon, args.almost_same)
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
            suptitile=f'{metadata.get("batch_name") or ""} {metadata.get("sub_batch_name") or ""} {golden_shape.C}x{golden_shape.H}x{golden_shape.W}',
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
        golden_range_min=golden_range_min,
        golden_range_max=golden_range_max,
        layout=args.layout,
        metadata=metadata
    )
