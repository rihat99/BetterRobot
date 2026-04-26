"""Advisory shape-annotation coverage for public callables.

Walks every public symbol in ``better_robot.__all__`` and reports the
fraction whose function signature uses one of the jaxtyping aliases from
``better_robot._typing``. Always passes — the coverage number is printed
so CI can track it over time.

See ``docs/conventions/testing.md §4.5`` (advisory mode).
"""

from __future__ import annotations

import importlib
import inspect
import typing

import better_robot

# Names from ``_typing`` are jaxtyping aliases; we count any annotation
# that *mentions* one of them.
_TYPING_ALIASES = {
    "SE3Tensor", "SO3Tensor", "Quaternion", "TangentSE3", "TangentSO3",
    "JointPoseStack", "FramePoseStack", "ConfigTensor", "VelocityTensor",
    "JointJacobian", "JointJacobianStack",
}


def test_shape_annotation_coverage_advisory() -> None:
    total = 0
    annotated = 0
    for name in better_robot.__all__:
        obj = getattr(better_robot, name)
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        total += 1
        ann = sig.return_annotation
        params = list(sig.parameters.values())
        all_annots = [str(ann)] + [str(p.annotation) for p in params]
        if any(alias in joined for alias in _TYPING_ALIASES for joined in all_annots):
            annotated += 1
    if total:
        ratio = annotated / total
        print(f"\nshape-annotation coverage: {annotated}/{total} = {ratio:.0%}")
    # Always passes — advisory.
