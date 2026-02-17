from argparse import ArgumentParser
from dataclasses import dataclass

from coordinates import TensorLayout


@dataclass
class Args:
    tensor_layout: TensorLayout

    epsilon: float
    """
    The maximum difference, in absolute values, after which two
    numbers are considered different
    """

    root_path: str
    """
    The path where all the fault injection experiments batches folders are
    stored
    """

    golden_path: str
    """
    The relative path from the test batch folder to the golden tensor of the batch
    """

    faulty_path: str
    """
    The relative path from the test batch folder to faulty tensor folders
    """

    output_dir: str
    """
    The absolute path where the output files of this program are stored
    """

    parallel: int
    """
    The number of parallel processes
    """

    @classmethod
    def from_argparse(cls, args):
        return Args(
            tensor_layout=TensorLayout.NCHW,
            epsilon=args.epsilon,
            root_path=args.root_path,
            golden_path=args.golden_path,
            faulty_path=args.faulty_path,
            output_dir=args.output_dir,
            parallel=args.parallel,
        )


def create_parser() -> ArgumentParser:
    """
    Sets up an argparse.ArgumentParser instance
    """
    parser = ArgumentParser(
        prog="Tensor Error Classifier",
        description="Compares Faulty Tensors with a golden one, and classifies them",
    )
    parser.add_argument("root_path", help="Path to the root folder of the test results")
    parser.add_argument("golden_path", help="Relative path (from root) to the golden file")
    parser.add_argument("faulty_path", help="Relative path (from root) to the folders containing faulty files")
    parser.add_argument("output_dir", help="Output directory to save the channel count reports")
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        help="Use N parallel processes",
        metavar="N",
        default=1,
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=1e-3,
        help="Value for equality tolerance",
    )

    return parser
