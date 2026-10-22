"""Periodic novelty maintenance over the unresolved pool. See README.md."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # also works without `pip install -e .`
from taxonomy_classifier.cli import novelty_main  # noqa: E402

if __name__ == "__main__":
    novelty_main()
