"""Fail unless every requirement of the installed taxonomy classifier, including its build extra, is met.

``pip check`` is not enough here: it reports conflicts anywhere in the environment (including ones unrelated to the
classifier) and ignores extras. The classifier is installed with ``--no-deps`` because poetry.lock pins its
dependencies, so this is what turns a drift between the lock and the package's ranges into a failed image build.
"""

import sys
from importlib import metadata

from packaging.requirements import Requirement

PACKAGE = "neo-taxonomy-classifier"
EXTRAS = ("", "build")


def main() -> int:
    problems = []
    for extra in EXTRAS:
        for raw in metadata.requires(PACKAGE) or []:
            req = Requirement(raw)
            if req.marker is not None and not req.marker.evaluate({"extra": extra}):
                continue
            if req.marker is not None and not extra and "extra" in str(req.marker):
                continue
            try:
                have = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                problems.append(f"{req.name} is not installed (needed: {req})")
                continue
            if not req.specifier.contains(have, prereleases=True):
                problems.append(f"{req.name} {have} is installed, the classifier needs {req.specifier}")
    for p in sorted(set(problems)):
        print(f"{PACKAGE}: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
