from collections import OrderedDict, defaultdict
import json
import os
from typing import Callable, Union
from coordinates import TensorLayout, map_to_coordinates
import numpy as np
import logging as log
from spatial_classifier import SpatialClass, accumulate_max
from tensor_analyzer import analyze_tensor


from utils import sort_dict, double_int_defaultdict, list_defaultdict


def analyze_tensor_directory(
    faulty_path: str,
    golden: np.ndarray,
    report_output_path: str,
    layout: TensorLayout,
    epsilon: int,
    almost_same: bool,
    image_output_dir: str = "",
    visualize_errors: bool = False,
    save_report: bool = True,
    metadata: dict = {},
    on_tensor_completed: Union[Callable[[], None], None] = None,
):
    golden_shape = map_to_coordinates(golden.shape, layout)

    # Get list of all file ending in .npy
    faulty_files_path = [
        os.path.join(faulty_path, entry)
        for entry in os.listdir(faulty_path)
        if entry.split(".")[1] == "npy"
    ]

    log.info(f"Found {len(faulty_files_path)} faulty tensors to analize")

    # Initialize data structures for reports
    sp_class_count = defaultdict(int)
    dom_class_count = defaultdict(int)
    tensor_report = OrderedDict()
    corrupt_cardinality = defaultdict(int)
    raveled_patterns = defaultdict(int)
    error_patterns = defaultdict(double_int_defaultdict)
    class_params = defaultdict(list_defaultdict)
    corrupted_values = 0

    # Iterate over all tensor files
    for file_path in sorted(faulty_files_path):
        sp_class, tensor_dict = analyze_tensor(
            file_path=file_path,
            golden=golden,
            layout=layout,
            epsilon=epsilon,
            almost_same=almost_same,
            visualize_errors=visualize_errors,
            output_dir=image_output_dir,
            metadata=metadata,
        )
        if on_tensor_completed is not None:
            on_tensor_completed()
        sp_class_count[sp_class] += 1

        if sp_class != "masked" and sp_class != "skipped":
            error_pattern = str(tensor_dict["error_pattern"])
            corr_val_count = tensor_dict["corrupted_values"]

            tensor_report[os.path.basename(file_path)] = tensor_dict
            raveled_patterns[str(tensor_dict["raveled_offsets"])] += 1
            corrupt_cardinality[tensor_dict["corrupted_values"]] += 1
            corrupted_values += tensor_dict["corrupted_values"]
            error_patterns[corr_val_count][sp_class][error_pattern] += 1
            class_params[corr_val_count][sp_class] = accumulate_max(
                class_params[corr_val_count][sp_class],
                tensor_dict["class_params"]["MAX"]
                if "MAX" in tensor_dict["class_params"]
                else [],
            )
            for dom_class, freq in tensor_dict["domain_classes"].items():
                dom_class_count[dom_class] += freq

    classified_tensors = (
        len(faulty_files_path) - sp_class_count["masked"] - sp_class_count["skipped"]
    )

    if classified_tensors == 0:
        log.warn(f"No tensors were classified in {faulty_path}")
        return {}
    # Main Report Generation
    main_report = OrderedDict()
    global_data = OrderedDict()

    global_data["tensor_shape"] = golden_shape._asdict()
    global_data["tensor_size"] = golden.size
    global_data["tensors"] = len(faulty_files_path)
    global_data["masked"] = sp_class_count["masked"]
    global_data["skipped"] = sp_class_count["skipped"]
    global_data["classified_tensors"] = classified_tensors
    global_data["corrupted_values"] = corrupted_values
    global_data["error_patterns"] = error_patterns
    global_data["class_params"] = class_params
    global_data["average_corrupted_values_pct"] = (
        float(corrupted_values) / golden.size / classified_tensors * 100
    )
    global_data["spatial_classes"] = sort_dict(
        {
            sp_class.display_name(): sp_class_count[sp_class.display_name()]
            for sp_class in SpatialClass
        }
    )
    global_data["spatial_classes_pct"] = sort_dict(
        {
            sp_class.display_name(): sp_class_count[sp_class.display_name()]
            for sp_class in SpatialClass
        }
    )

    global_data["domain_classes"] = sort_dict(dom_class_count)
    global_data["domain_classes_pct"] = sort_dict(
        {
            dom_class: float(freq) / golden.size / classified_tensors * 100
            for dom_class, freq in dom_class_count.items()
        }
    )
    global_data["corrupt_cardinality"] = sort_dict(corrupt_cardinality)

    global_data["corrupt_cardinality_pct"] = (
        {
            n_errors: freq / global_data["corrupted_values"] * 100
            for n_errors, freq in corrupt_cardinality.items()
        },
    )
    global_data["raveled_patterns"] = OrderedDict(
        sorted(
            [
                (pattern, freq)
                for pattern, freq in raveled_patterns.items()
                if freq > 5 / 100 * len(raveled_patterns)
            ],
            key=lambda x: x[1],
            reverse=True,
        )
    )
    main_report["metadata"] = metadata
    main_report["global_data"] = global_data
    main_report["tensors"] = tensor_report

    if save_report:
        report_path = report_output_path

        with open(report_path, "w") as f:
            f.writelines(json.dumps(main_report, indent=2))

    return main_report
