"""A stand-in for the classifier package with the interface Airweave calls (no behaviour)."""

import sys
import types

from airweave.domains.taxonomy.package import JOBS_INTERFACE, SYNC_INTERFACE


def stub_classifier_package(monkeypatch, *, without=()):
    """Put a module for every name of the sync and jobs interfaces into ``sys.modules``.

    ``without`` leaves out modules (``"taxonomy_classifier.contract"``) or names
    (``"taxonomy_classifier.contract.to_contract"``), to see the check refuse them.
    """
    modules = {}
    for interface in (SYNC_INTERFACE, JOBS_INTERFACE):
        for module_name, names in interface.items():
            parts = module_name.split(".")
            for i in range(1, len(parts) + 1):
                modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
            for name in names:
                if f"{module_name}.{name}" not in without:
                    setattr(modules[module_name], name, object())
    for module_name, module in modules.items():
        if module_name in without:
            monkeypatch.setitem(sys.modules, module_name, None)  # import raises ImportError
        else:
            monkeypatch.setitem(sys.modules, module_name, module)
