"""SolidWorks stream parser used by the swx2blend Blender add-on."""

from .convert import ConvertPart, ConvertResult, convert_solidworks
from .parser import parse_solidworks_file

__all__ = ["ConvertPart", "ConvertResult", "convert_solidworks", "parse_solidworks_file"]
