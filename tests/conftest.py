"""
Test-collection shim: prevents importing the top-level draco/__init__.py
(which pulls in torch, a large optional dependency not installed in this
environment) so the .draco format / CPU quantization / CPU dispatcher
modules — which do not depend on torch — can be tested in isolation.

This only affects test collection; it does not change draco's real
package behavior for normal (non-test) imports.
"""

import importlib.util
import os
import sys
import types

_draco_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "draco")


def _stub_package(dotted_name: str, subdir: str) -> None:
    if dotted_name not in sys.modules:
        mod = types.ModuleType(dotted_name)
        mod.__path__ = [os.path.join(_draco_dir, subdir)]
        sys.modules[dotted_name] = mod


_stub_package("draco", "")
_stub_package("draco.quantization", "quantization")
_stub_package("draco.runtime", "runtime")
_stub_package("draco.utils", "utils")
