from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
CONSENSUS_LAB = ROOT / 'consensus_lab'
if str(CONSENSUS_LAB) not in sys.path:
    sys.path.insert(0, str(CONSENSUS_LAB))
