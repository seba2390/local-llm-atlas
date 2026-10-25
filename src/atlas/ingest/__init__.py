"""Importers that pull published data into the atlas.

Each importer is a module in this package exposing `run() -> dict`. Register it in IMPORTERS.
"""

import importlib

IMPORTERS = [
    "epoch",
]


def run(name: str) -> dict:
    return importlib.import_module(f"{__name__}.{name.replace('-', '_')}").run()
