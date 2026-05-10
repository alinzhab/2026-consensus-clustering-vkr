"""Общая фикстура для тестов.

Делает `consensus_lab/` импортируемым по короткому имени модулей,
чтобы тесты могли писать `from sdgca import run_sdgca` без переустановки
пакета. Это согласуется с тем, как существующий код проекта устроен —
в `app.py` тоже используется `sys.path.insert(0, CONSENSUS_LAB_DIR)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSENSUS_LAB = ROOT / "consensus_lab"
if str(CONSENSUS_LAB) not in sys.path:
    sys.path.insert(0, str(CONSENSUS_LAB))
