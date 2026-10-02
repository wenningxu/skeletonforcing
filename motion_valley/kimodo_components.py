"""Load pinned, unmodified Kimodo network modules without the demo/LLM imports.

Only package initializers are bypassed; backbone and two-stage forward code are
executed directly from the official files. No installed Kimodo is substituted.
"""
import importlib
from pathlib import Path
import sys
import types


def official_twostage():
    root = Path(__file__).resolve().parents[1] / 'third_party/kimodo/kimodo'
    for name, folder in [('kimodo', root), ('kimodo.model', root / 'model')]:
        if name in sys.modules:
            paths = list(getattr(sys.modules[name], '__path__', []))
            if str(folder) not in paths:
                raise RuntimeError(f'{name} is already loaded from another location')
        else:
            module = types.ModuleType(name)
            module.__path__ = [str(folder)]
            module.__package__ = name
            sys.modules[name] = module
    return importlib.import_module('kimodo.model.twostage_denoiser').TwostageDenoiser
