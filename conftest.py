"""Makes the `lossysplat` package importable when pytest is run from the repository root."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
