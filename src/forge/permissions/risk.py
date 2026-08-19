"""Risk classification — the vocabulary shared by tools and the permission policy.

This lives in its own dependency-free module on purpose. ``forge.tools`` and
``forge.permissions`` both need it: a tool declares its risk, and the policy
decides what that risk is allowed to do. Defining it in either package would
make the two import each other in a cycle, so it sits below both.

Import it from :mod:`forge.tools.base` (where it reads naturally alongside
``Tool``) or from here; they are the same object.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Risk"]


class Risk(StrEnum):
    """How much damage a tool call can do, which drives the approval decision.

    ``READ``        observes only — always safe to run unattended.
    ``WRITE``       creates or modifies things the agent can also inspect and fix.
    ``DESTRUCTIVE`` discards work or reaches outside the workspace; needs a human.
    """

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
