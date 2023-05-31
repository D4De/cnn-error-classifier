from enum import Enum
import os



class SpatialClass(Enum):
    SAME = 0
    SINGLE = 1
    SAME_COLUMN = 2
    SAME_ROW = 3
    SINGLE_CHANNEL_RANDOM = 4
    BULLET_WAKE = 5
    SHATTERED_GLASS = 6
    QUASI_SHATTERED_GLASS = 7
    SKIP_4 = 8
    CHANNEL_ALIGNED_SAME_BLOCK = 9
    MULTIPLE_CHANNELS_UNCATEGORIZED = 10
    CHANNEL_ALIGNED_SINGLE_BLOCK = 11
    TENSOR_ALIGNED_SINGLE_BLOCK = 12
    SINGLE_BLOCK = 13
    MULTI_CHANNEL_BLOCK = 14
    SHATTERED_CHANNEL = 15
    QUASI_SHATTERED_CHANNEL = 16
    SINGLE_CHANNEL_ALTERNATED_BLOCKS = 17
    SKIP_2 = 18
    FULL_CHANNELS = 19
    RECTANGLES = 20

    def display_name(self) -> str:
        """
        Returns the display name of the Spatial Class. The Display is the name of the class in snake case
        """
        return self.name.lower()

    def class_folder(self, output_path) -> str:
        """
        Returns the path of the subfolder (inside output_path) where the visualizations of tensors classified under this class are stored
        """
        return os.path.join(output_path, self.display_name())

    def output_path(self, output_path, basename) -> str:
        return os.path.join(output_path, self.display_name(), basename)
    

def to_classes_id(name) -> str:
    """
    Maps a SpatialClass enum instance to his id used in the CLASSES framework.
    """
    if name == SpatialClass.SAME_ROW.display_name():
        return "0"
    elif name == SpatialClass.SAME_COLUMN.display_name():
        return "1"
    elif name == SpatialClass.SINGLE_MAP_RANDOM.display_name():
        return "3"
    elif name == SpatialClass.BULLET_WAKE.display_name():
        return "4"
    elif name == SpatialClass.SHATTERED_GLASS.display_name():
        return "6"
    elif name == SpatialClass.QUASI_SHATTERED_GLASS.display_name():
        return "7"
    elif name == SpatialClass.SKIP_4.display_name():
        return "1001"
    elif (
        name == SpatialClass.CHANNEL_ALIGNED_SAME_BLOCK.display_name()
        or name == SpatialClass.CHANNEL_ALIGNED_SINGLE_BLOCK.display_name()
    ):
        return "1002"
    elif name == SpatialClass.MULTIPLE_CHANNELS_UNCATEGORIZED.display_name():
        return "8"
    elif name == SpatialClass.TENSOR_ALIGNED_SINGLE_BLOCK.display_name():
        return "1003"
    elif name == SpatialClass.SINGLE_BLOCK.display_name():
        return "1004"
    elif name == SpatialClass.MULTI_CHANNEL_BLOCK.display_name():
        return "1005"
    elif name == SpatialClass.SHATTERED_CHANNEL.display_name():
        return "1006"
    elif name == SpatialClass.QUASI_SHATTERED_CHANNEL.display_name():
        return "1007"
    elif name == SpatialClass.SINGLE_CHANNEL_ALTERNATED_BLOCKS.display_name():
        return "1008"
    elif name == SpatialClass.SKIP_2.display_name():
        return "1009"
    elif name == SpatialClass.FULL_CHANNELS.display_name():
        return "1010"
    elif name == SpatialClass.RECTANGLES.display_name():
        return "1011"