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

def report_uncategorized_tensors(unit_dir: str, analyzed_tensors: list):
    report_path = os.path.join(unit_dir, 'uncategorized_log.csv')

    with open(report_path, 'w', newline='') as csvlog:
        logwriter = csv.writer(csvlog)
        logwriter.writerow(['Type', 'Error Number', 'Injection Number'])

        for tensor in analyzed_tensors:
            if tensor.spatial_class == SpatialClass.SINGLE_CHANNEL_RANDOM:
                logwriter.writerow(['Single', tensor.error_number, str(tensor.injection_number)])
            elif tensor.spatial_class == SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED:
                logwriter.writerow(['Multiple', tensor.error_number, str(tensor.injection_number)])
