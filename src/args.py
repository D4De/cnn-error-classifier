from __future__ import annotations
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
import os
from typing import Tuple, Union
from coordinates import TensorLayout


@dataclass
class Args:
    """
    Typed holder of arguments received from command line and parsed using argparse
    """

    layout: TensorLayout
    """
    Holds how the tensor is structured
    """

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

    limit: Union[int, None]
    """
    The maximum number of test batches to analyze
    """

    visualize: bool
    """
    If true, the user requested to generate and save images for
    visualizing the errors
    """

    almost_same: bool
    """
    If true, the "ALMOST_SAME" domain class is enabled. All erroneus values
    of which the absolute value difference with the respective golden value is less
    than epsilon will be classified as "ALMOST_SAME". If false "ALMOST_SAME" class
    is collapsed with "SAME" (no error)
    """

    partial_reports: bool
    """
    If true, the reports for each sub_batch will be generated
    """

    classes: Union[Tuple[str, str], None]
    """
    If not none, the user requested to export classes error models files.
    The tuple of two elements contains the name of the classes file.
    """

    visualize_path: str
    """
    The path where the visualizations of the errors will be saved (if visualize is true)
    """

    reports_path: str
    """
    The path where the partial reports will be saved (if partial_report is true)
    """

    parallel: int
    """
    The number of parallel processes
    """

    database: bool
    """
    Store experiment data in a sqlite database file
    """

    classes_category_absolute_cutoff : int
    """
    
    """

    classes_category_relative_cutoff : float
    """
    """

    @classmethod
    def from_argparse(cls, args: Namespace) -> Args:
        """
        Generate an instance of Args from the argument parsed by argparse
        """
        if args.nhwc:
            tensor_layout = TensorLayout.NHWC
        else:
            tensor_layout = TensorLayout.NCHW
        return Args(
            tensor_layout,
            epsilon=args.epsilon,
            root_path=args.root_path,
            golden_path=args.golden_path,
            faulty_path=args.faulty_path,
            output_dir=args.output_dir,
            limit=args.limit,
            visualize=args.visualize,
            almost_same=args.almost_same,
            partial_reports=args.partial_reports,
            classes=args.classes,
            visualize_path=os.path.join(args.output_dir, "visualize"),
            reports_path=os.path.join(args.output_dir, "reports"),
            parallel=args.parallel,
            database=args.database,
            classes_category_absolute_cutoff=5,
            classes_category_relative_cutoff=0.01,            
        )


def create_parser() -> ArgumentParser:
    """
    Sets up an argparse.ArgumentParser instance
    """
    parser = ArgumentParser(
        prog="Tensor Error Classifier",
        description="Compares Faulty Tensors with a golden one, and classifies them",
    )
    parser.add_argument(
        "root_path", help="A path to the root folder of the test results"
    )
    parser.add_argument(
        "golden_path",
        help="A relative path that reaches the golden file from the test batch home folder",
    )
    parser.add_argument(
        "faulty_path",
        help="A relative path that reaches the folders containing faulty files from the test batch home folder",
    )
    parser.add_argument(
        "output_dir", help="Path where to write the ouput of the analysis"
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Limit the number of tensor to process",
        metavar="N",
    )
    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        help="Use N parallel processes",
        metavar="N",
        default=1,
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Generate images that show visually the differences between tensors (overwriting the image generated before)",
    )

    parser.add_argument(
        "-as",
        "--almost-same",
        action="store_true",
        help="Include in the plot the values that are very close to golden value (< EPS)",
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=1e-3,
        help="Set epsilon value. Differences below epsilon are treated as almost same value and are not plotted (unless -as is enabled)",
    )
    parser.add_argument(
        "-pr",
        "--partial-reports",
        action="store_true",
        help="Generate partial reports",
    )
    parser.add_argument(
        "--classes",
        nargs=2,
        metavar=("Sx", "OPERATION"),
        help="Generate models for classes",
    )

    parser.add_argument(
        "-db",
        "--database",
        action="store_true",
        help="Store results in a sqlite database",
    )
    tensor_format_group = parser.add_mutually_exclusive_group()
    tensor_format_group.add_argument(
        "-nchw",
        action="store_true",
        help="Loaded tensors are stored using NCHW dimensional order (default)",
    )
    tensor_format_group.add_argument(
        "-nhwc",
        action="store_true",
        help="Loaded tensors are stored using NHWC dimensional order",
    )
    return parser
