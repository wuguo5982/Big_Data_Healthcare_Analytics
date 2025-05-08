import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
VERSION_PATH = PACKAGE_ROOT / "VERSION"

with open(VERSION_PATH) as f:
    __version__ = f.read().strip()
