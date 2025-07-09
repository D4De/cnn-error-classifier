import os
import csv
import json

from statistics import mean, stdev
from collections import OrderedDict
from aggregators import cardinalities_counts_by_sp_class, spatial_classes_counts, tensor_count_by_shape
from analyzed_tensor import AnalyzedTensorConv, AnalyzedTensorFC
from spatial_classifier.spatial_class import SpatialClass


def generate_unit_report_conv(unit_dir: str, analyzed_tensors: list[AnalyzedTensorConv]):
    report = OrderedDict()

    report["classified_tensors"] = len(analyzed_tensors)
    report["tensors_by_shape"] = tensor_count_by_shape(analyzed_tensors)
    report["spatial_classes"] = spatial_classes_counts(analyzed_tensors)
    report["class_cardinalites"] = cardinalities_counts_by_sp_class(analyzed_tensors)

    report_path = os.path.join(unit_dir, 'unit_report.json')
    with open(report_path, 'w') as rf:
        json.dump(report, rf, indent=2)


def generate_unit_report_fc(unit_dir: str, analyzed_tensors: list[AnalyzedTensorFC]):
    report = OrderedDict()

    report["classified_tensors"] = len(analyzed_tensors)
    report["tensor_shape"] = analyzed_tensors[0].shape

    # determine min, max, avg and std. dev. of number of corrupted values, L1 distance and L2 distance
    corrupted_values_counts = []
    L1_distances = []
    L2_distances = []

    for tensor in analyzed_tensors:
        corrupted_values_counts.append(tensor.corrupted_values_count)
        L1_distances.append(tensor.L1_distance)
        L2_distances.append(tensor.L2_distance)

    report["num_corrupted_values"] = {
        "min": min(corrupted_values_counts),
        "max": max(corrupted_values_counts),
        "mean": mean(corrupted_values_counts),
        "stdev": stdev(corrupted_values_counts)
    }
    report["L1_distance"] = {
        "min": min(L1_distances),
        "max": max(L1_distances),
        "mean": mean(L1_distances),
        "stdev": stdev(L1_distances)
    }
    report["L2_distance"] = {
        "min": min(L2_distances),
        "max": max(L2_distances),
        "mean": mean(L2_distances),
        "stdev": stdev(L2_distances)
    }

    report_path = os.path.join(unit_dir, 'unit_report.json')
    with open(report_path, 'w') as rf:
        json.dump(report, rf, indent=2)


def report_tensor_results(output_dir: str, analyzed_tensors: list[AnalyzedTensorConv]):
    """
    Writes a csv report concerning all analyzed tensors.
    The report can be later used to create the error tensor visualizations for all spatial classes
    or for a specific subset of them (e.g. only the uncategorized tensors).
    """
    report_path = os.path.join(output_dir, 'tensor_results_report.csv')

    with open(report_path, 'w', newline='') as csvlog:
        logwriter = csv.writer(csvlog)
        logwriter.writerow(['Spatial Class', 'Unit Group', 'HW Unit', 'Error Number', 'Injection Number', 'Corrupted Channels'])

        for tensor in analyzed_tensors:
            logwriter.writerow([
                tensor.spatial_class.display_name(),
                tensor.group,
                tensor.hw_unit,
                tensor.error_number,
                str(tensor.injection_number),
                str(tensor.corrupted_channels)
            ])
