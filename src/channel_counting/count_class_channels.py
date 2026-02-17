import os
import sys

from tqdm import tqdm
from queue import Empty
from functools import partial
from multiprocessing import Manager, Pool, Process, Queue

from utils import read_npz_sizes
from channel_counting.batch_counter import analyze_batch
from channel_counting.args import Args, create_parser


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


def main():
    # Parsing command line arguments
    parser = create_parser()
    argparse_args = parser.parse_args()
    # The args variable contains the configuration of this application given by the user via CLI arguments
    args = Args.from_argparse(argparse_args)

    # Create additional output directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ctrl'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'data'), exist_ok=True)

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
        result_value = result.get()
    progress_process.join()


if __name__ == "__main__":
    main()