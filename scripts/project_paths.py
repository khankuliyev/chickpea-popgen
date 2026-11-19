"""File locations shared by the analysis scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TREES = ROOT / "results" / "trees"
