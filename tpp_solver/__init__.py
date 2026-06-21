"""TPP Solver: Thermal Proteome Profiling analysis package."""
import os

__version__ = "1.2.3"

# Repository root (one level above this package). Bundled data files such as the
# GO database and the example TSV/CSV live here, so paths must be resolved
# relative to this rather than to a module inside the package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
