"""My Very Small Language Model - Playing around with Large Language Models."""

__version__ = "0.1.0"
__author__ = "John Norrie"
__email__ = "norrie2007@gmail.com"

# Import main classes/functions here when they're created
from .data_loader import EmailDataLoader

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "EmailDataLoader"
]
