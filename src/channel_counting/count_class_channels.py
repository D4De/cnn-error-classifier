import os
import sys
import csv

from tqdm import tqdm
from queue import Empty
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from utils import read_npz_sizes
from channel_counting.args import Args, create_parser
from channel_counting.batch_counter import analyze_batch, output_dir_from_input_dir
from channel_counting.classify import SPATIAL_CLASS_NAMES

def precalculate_workload(
    hw_unit_paths: list[str], errors_filename: str = 'errors.npz'
):
    """
    Calculates the number of tensors to analyze in the subdirectories. Returns a tuple containing the total number of tensors in all
    subdirectories and a dict that contains the number of tensors for each one.
    """
    if not errors_filename.endswith('.npz'):
        errors_filename = errors_filename + '.npz'

    total_tensors = 0
    sizes = {}

    for unit_path in hw_unit_paths:
        num_tensors = 0

        path_to_errors = os.path.join(unit_path, errors_filename)

        for tensor_shape in read_npz_sizes(path_to_errors):
            # tensor_shape here is a 5-uple with this interpretation: (num_injections, num_batches, num_channels, height, width)
            # so, the amount of "single" output tensors for each file in the npz archive is injection_number * batch_number
            num_tensors += (tensor_shape[0] * tensor_shape[1])

        print(f'\t{unit_path}: found {num_tensors} tensors to analyze')

        sizes[unit_path] = num_tensors
        total_tensors += num_tensors

    return total_tensors, sizes


def progress_handler(queue: Queue, work: int):
    """
    A process that updates the progress bar.
    The working threads send a "processed" message to this thread, using the queue, whenever they complete processing a tensor.
    The responsibility of this process is to update the progress bar when a "processed" is received.
    """
    with tqdm(total=work) as pbar:
        work_count = 0
        while True:
            try:
                message, args = queue.get(True, 15)
                if message == "exit":
                    break
                elif message == "processed":
                    work_count += args
                    pbar.update(args)
                    sys.stdout.flush()
                    if work_count >= work:
                        print("Work complete!")
                        break
            except Empty:
                print("No updates received. Quitting")
                break


def aggregate_unit_results(unit_dir: str):
    class_counts: dict[str, int] = {}

    # initialize dictionary: both single and multi channels for each class
    for class_name in SPATIAL_CLASS_NAMES:
        class_counts[class_name + '_single'] = 0
        class_counts[class_name + '_multi']  = 0

    # count occurrences of single/multi channel for each class in the file
    in_csv_path = os.path.join(unit_dir, 'channel_counts.csv')
    with open(in_csv_path) as f:
        csvreader = csv.DictReader(f)
        for row in csvreader:
            class_name = row['spatial_class']
            channel_count = int(row['corrupted_channels'])

            if channel_count == 1:
                class_counts[class_name + '_single'] += 1
            else:
                class_counts[class_name + '_multi'] += 1

    
    # compute frequencies and save to file
    out_csv_path = os.path.join(unit_dir, 'class_frequencies.csv')
    fieldnames = ['spatial_class', 'channel_type', 'frequency']
    with open(out_csv_path, 'w', newline='') as f:
        csvwriter = csv.DictWriter(f, fieldnames=fieldnames)
        csvwriter.writeheader()

        for class_name in SPATIAL_CLASS_NAMES:
            class_total = class_counts[class_name + '_single'] + class_counts[class_name + '_multi']

            if class_total == 0:
                # if a class is absent, arbitrarily set both to -1
                single_freq = -1.0
                multi_freq  = -1.0
            else:
                single_freq = float(class_counts[class_name + '_single'] / class_total)
                multi_freq  = float(class_counts[class_name + '_multi'] / class_total)
                
            csvwriter.writerow({'spatial_class': class_name, 'channel_type': 'single', 'frequency': single_freq})
            csvwriter.writerow({'spatial_class': class_name, 'channel_type': 'multi', 'frequency': multi_freq})


def main():
    # Parsing command line arguments
    parser = create_parser()
    argparse_args = parser.parse_args()
    # The args variable contains the configuration of this application given by the user via CLI arguments
    args = Args.from_argparse(argparse_args)

    # Create additional output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    # The error batches are grouped wrt the hardware path for the injection (control or data)
    ctrl_dir = os.path.join(args.root_path, 'ctrl')
    data_dir = os.path.join(args.root_path, 'data')

    if os.path.isdir(ctrl_dir):
        print('Found ctrl directory')
        ctrl_dir_names = [dir for dir in os.listdir(ctrl_dir) if os.path.isdir(os.path.join(ctrl_dir, dir))]
    else:
        print(f'Control path directory {ctrl_dir} is missing.')

    if os.path.isdir(data_dir):
        print('Found data directory')
        data_dir_names = [dir for dir in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir))]
    else:
        print(f'Data path directory {data_dir} is missing.')

    hw_unit_dirs = \
        [os.path.join(ctrl_dir, dir_name) for dir_name in ctrl_dir_names] + \
        [os.path.join(data_dir, dir_name) for dir_name in data_dir_names] 
    print(f"Found {len(hw_unit_dirs)} hardware units directories to analyze.")

    # workload == total number of tensors to analyze (for progress bar)
    total_tensors, unit_dirs_sizes = precalculate_workload(hw_unit_dirs)
    print(f"Found {total_tensors} total tensors to analyze")


    # Multiprocessing setup
    manager = Manager()
    progress_queue = manager.Queue()
    # Pre-Configure the arguments of analyze_batch that are the same for all batches
    # batch_partial accepts only one parameter
    batch_partial = partial(analyze_batch, args=args, queue=progress_queue)

    progress_process = Process(
        target=progress_handler, args=(progress_queue, total_tensors)
    )
    # Start the progress bar process
    progress_process.start()
    # Start the worker processes
    with Pool(args.parallel) as pool:
        result = pool.map_async(batch_partial, hw_unit_dirs, chunksize=1)
        _ = result.get()
    progress_process.join()

    # aggregate results
    for unit_dir in hw_unit_dirs:
        out_dir = output_dir_from_input_dir(args.output_dir, unit_dir)
        aggregate_unit_results(out_dir)


if __name__ == "__main__":
    main()