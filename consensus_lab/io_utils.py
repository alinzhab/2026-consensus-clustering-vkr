from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def append_tsv_row(path: str | Path, row: Mapping, header: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(header), delimiter='\t')
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in header})


def write_tsv(path: str | Path, rows: Iterable[Mapping], header: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(header), delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in header})


def read_tsv(path: str | Path) -> list[dict]:
    path = Path(path)
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))
