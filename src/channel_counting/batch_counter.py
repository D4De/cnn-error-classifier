import os
import csv
import numpy as np

from multiprocessing import Queue

from channel_counting.args import Args
from coordinates import map_to_coordinates
from channel_counting.spatial_classifier import spatial_classification


def analyze_batch(in_out_dirs: tuple[str, str], args: Args, queue: Queue | None):
    """
    Produces a csv file listing, for each tensor in the batch, the spatial class and the number of corrupted channels.
    """
    in_dir, out_dir = in_out_dirs

    topdir_name = os.path.basename(os.path.dirname(in_dir))
    batch_name = topdir_name + '_' + os.path.basename(in_dir)

    golden_path = os.path.join(in_dir, args.golden_path)

    # check if the errors archive exists
    errors_path = os.path.join(in_dir, args.faulty_path)

    if not os.path.exists(errors_path):
        print(f"Skipping {batch_name} batch. Errors archive not found.")
        return None
            

    # if there is a queue specified prepare the lambda for signalling to the progress bar process that a tensor was processed
    if queue is not None:
        on_tensor_completed = lambda: queue.put(("processed", 1), block=False)
    else:
        on_tensor_completed = None


    # check if the golden file exists
    if not os.path.exists(golden_path):
        print(f"Skipping {batch_name} since it does not contain a golden file.")
        return None

    # load golden file for the batch
    try:
        golden: np.ndarray = np.load(golden_path)
    except:
        print(f"Skipping {batch_name} batch. Could not read golden file.")
        return None

    if golden is None:
        print(f'Skipping {batch_name} batch. Golden file is malformed or empty.')
        return None

    # determine type of layer according to golden shape
    if len(golden.shape) != 4:
        print(f"Skipping {batch_name} batch. Dimension of golden not supported: {golden.shape}.")
        return None
    
    # prepare channel count csv file
    with open(os.path.join(out_dir, 'channel_counts.csv'), 'w', newline='') as csvfile:
        fieldnames = ['spatial_class', 'corrupted_channels']
        csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        csvwriter.writeheader()

        analyze_errors_archive(
            errors_path,
            golden,
            args,
            on_tensor_completed=on_tensor_completed,
            csvwriter=csvwriter,
        )


# ARCHIVE ANALYSIS ---------------------------------------------------------------------------------------------------

def analyze_errors_archive(
        errors_path: str,
        golden: np.ndarray,
        args: Args,
        on_tensor_completed,
        csvwriter,
):
    # load the npz archive
    errors_archive = np.load(errors_path)

    # iterate over the files in the archive
    for error_number, error in errors_archive.items():
        golden_tensor = golden[int(error_number)]
        # each file is a 5D tensor: iterate twice to get a single tensor
        for injection_number, injection in enumerate(error):
            for error_tensor in injection:
                analyze_error_tensor(
                    errors_path=errors_path,
                    error_number=error_number,
                    injection_number=injection_number,
                    tensor=error_tensor[np.newaxis, :], #reshape to 4D
                    golden=golden_tensor[np.newaxis, :], #reshape to 4D
                    args=args,
                    csvwriter=csvwriter,
                )

                if on_tensor_completed is not None:
                    on_tensor_completed()


# SINGLE TENSOR ANALYSIS ---------------------------------------------------------------------------------------------

def analyze_error_tensor(
    errors_path      : str,
    error_number     : str,
    injection_number : int,
    tensor           : np.ndarray,
    golden           : np.ndarray,
    args             : Args,
    csvwriter,
):
    # Ensure that the shapes match
    if tensor.shape != golden.shape:
        print(f"Skipping {errors_path} number {error_number} injection {injection_number}." \
                f"Invalid shape (Faulty has shape: {tensor.shape}, Golden has shape: {golden.shape})")
        return None

    # A Coordinates object is a named tuple with fields N, H, W, C. The tensor shapes are transformed to fit a specific layout,
    # such as NCHW for PyTorch.
    error_shape  = map_to_coordinates(tensor.shape, args.tensor_layout)

    # Determine spots where the golden tensor and the error tensor differ
    diff_mask = np.abs(tensor - golden) >= args.epsilon

    # No diff = masked
    if np.count_nonzero(diff_mask) == 0:
        print(f"{errors_path} number {error_number} injection {injection_number} has no diffs with golden")
        return None

    # Perform spatial classifcation
    spatial_class, _, faulty_channels = spatial_classification(diff_mask, error_shape, args.tensor_layout)
    csvwriter.writerow({'spatial_class': spatial_class.display_name(), 'corrupted_channels': len(faulty_channels)})