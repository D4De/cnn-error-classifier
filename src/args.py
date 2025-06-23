from __future__ import annotations
import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass


@dataclass
class Args:
    """
    Typed holder of arguments received from command line.
    """

    # Required
    root_path: str
    """
    Path to the folder storing the results of an injection campaign for a single network layer.
    """

    output_dir: str
    """
    Path to the output folder for the analysis. The folder does not need to exist.
    """

    # Optional
    epsilon: float
    """
    Two numbers are considered actually different if they differ by at least this value.
    Default is 1e-3.
    """

    golden_path: str
    """
    Relative path to the golden file from the root path. Default is 'golden.npy'. If a different file name
    was used, specify it with this.
    """

    errors_archive_filename: str
    """
    Each hardware unit directory should contain an 'errors.npz' archive containing the corrupted tensors.
    If a different file name was used, specify it with this.
    """

    visualize: bool
    """
    If true, visualizations of the error spatial patterns are generated. Default is false.
    """

    almost_same: bool
    """
    If true, the "ALMOST_SAME" domain class is enabled. All erroneous values
    of which the absolute value difference with the respective golden value is less
    than epsilon will be classified as "ALMOST_SAME". If false "ALMOST_SAME" class
    is collapsed with "SAME" (no error).
    Default is false.
    """

    classes: str | None
    """
    If not None, the CLASSES error models are generated. The value of this parameter is
    used as the name for the overall model.
    """
    
    parallel: int
    """
    Number of threads to spawn for the analysis. Default is 1.
    """

    database: bool
    """
    Store experiment data in a sqlite database file. Default is false.
    """

    visualize_limit : int
    """
    Maximum amount of error visualizations to generate for each hardware unit.
    Default is 0, meaning no limit.
    """

    # Derived or preconfigured
    classes_output_dir: str | None

    visualize_path: str | None

    classes_category_absolute_cutoff : int

    classes_category_relative_cutoff : float
    

    @classmethod
    def from_argparse(cls, args: Namespace) -> Args:
        """
        Generate an instance of Args from the arguments parsed by argparse.
        """
        return Args(
            epsilon=args.epsilon,
            root_path=os.path.realpath(args.root_path),
            golden_path=args.golden_path,
            errors_archive_filename=args.errors_archive_filename,
            output_dir=os.path.realpath(args.output_dir),
            visualize=args.visualize,
            almost_same=args.almost_same,
            classes=args.classes,
            parallel=args.parallel,
            database=args.database,
            visualize_limit=args.visualize_limit,
            classes_output_dir=os.path.join(args.output_dir, 'classes') if args.classes else None,
            visualize_path=os.path.join(args.output_dir, "visualize") if args.visualize else None,
            classes_category_absolute_cutoff=5,
            classes_category_relative_cutoff=0.01,
        )


def create_parser() -> ArgumentParser:
    """
    Sets up an argparse.ArgumentParser instance
    """
    parser = ArgumentParser(
        prog="main_nvdla.py",
        description="Compares the faulty tensors obtained from an injection campaign with the golden tensor and classifies them, producing " \
            "error models for CLASSES.",
    )
    parser.add_argument(
        "root_path",
        help="A path to the root folder of the test results for one operator."
    )
    parser.add_argument(
        "output_dir",
        help="Path to the output directory. The directory does not need to exist."
    )

    parser.add_argument(
        "--golden_path",
        help="A relative path to the golden file from the root path.",
        required=False,
        default='golden.npy'
    )
    parser.add_argument(
        "--errors_archive_filename",
        help="Name of the errors archive within each hardware unit directory.",
        required=False,
        default='errors.npz'
    )

    parser.add_argument(
        "-p",
        "--parallel",
        type=int,
        help="Use N parallel threads.",
        metavar="N",
        default=1,
    )
    parser.add_argument(
        "-v",
        "--visualize",
        action="store_true",
        help="Generate images showing the differences between tensors and the spatial patterns.",
    )

    parser.add_argument(
        "-as",
        "--almost-same",
        action="store_true",
        help="Include in the plot the values that are very close to golden value (< EPS).",
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=1e-3,
        help="Set epsilon value. Differences below epsilon are treated as almost same value and are not plotted (unless -as is enabled).",
    )

    parser.add_argument(
        "--classes",
        nargs=1,
        metavar=("MODEL_NAME"),
        help="Generate models for CLASSES.",
    )

    parser.add_argument(
        "-db",
        "--database",
        action="store_true",
        help="Store results in a sqlite database.",
    )
    parser.add_argument(
        "-vl",
        "--visualize-limit",
        type=int,
        default=0,
        help="Maximum number of visualized tensors per hardware unit."
    )

    return parser
