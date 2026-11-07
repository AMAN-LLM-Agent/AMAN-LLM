import sys
from pathlib import Path

# Make the `src` package (containing layer3/*) importable regardless of the
# directory pytest is invoked from.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
