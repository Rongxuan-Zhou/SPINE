from pathlib import Path
import sys

# Allow running tests without installing the package in editable mode.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
