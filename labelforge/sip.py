"""Compatibility shim for PyInstaller's PyQt5 hook.

PyQt5 provides SIP as ``PyQt5.sip`` in this environment, while PyInstaller's
standard PyQt5 hook still checks the legacy top-level ``sip`` module. Keeping
this shim avoids the spurious build warning without changing runtime behavior.
"""

from PyQt5.sip import *  # noqa: F401,F403
