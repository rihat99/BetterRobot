"""Every public ``lie`` function checks its tensor inputs at its boundary (principle 2.8, C-ERR-1).

The table lists the trailing shape of each tensor parameter; it must cover ``__all__`` of ``so3``, ``se3`` and
``tangents`` exactly, so a new public function cannot skip the check unseen. ``tests/support/boundary.py`` holds the
checks each entry runs.
"""

from __future__ import annotations

import inspect

from better_robot.lie import se3, so3, tangents
from tests.support.boundary import check_boundary
from tests.support.cases import failures

#: ``module.function`` -> trailing shape of each tensor parameter, in order (``()`` takes any shape).
PARAMETERS: dict[str, tuple[tuple[int, ...], ...]] = {
    "so3.act": ((4,), (3,)),
    "so3.compose": ((4,), (4,)),
    "so3.exp": ((3,),),
    "so3.from_axis_angle": ((3,), ()),
    "so3.from_euler": ((3,),),
    "so3.from_matrix": ((3, 3),),
    "so3.inverse": ((4,),),
    "so3.log": ((4,),),
    "so3.normalize": ((4,),),
    "so3.slerp": ((4,), (4,), ()),
    "so3.to_euler": ((4,),),
    "so3.to_matrix": ((4,),),
    "se3.act": ((7,), (3,)),
    "se3.adjoint": ((7,),),
    "se3.adjoint_inverse": ((7,),),
    "se3.compose": ((7,), (7,)),
    "se3.exp": ((6,),),
    "se3.from_axis_angle": ((3,), ()),
    "se3.from_matrix": ((4, 4),),
    "se3.from_translation": ((3,),),
    "se3.inverse": ((7,),),
    "se3.log": ((7,),),
    "se3.normalize": ((7,),),
    "se3.sclerp": ((7,), (7,), ()),
    "se3.to_matrix": ((7,),),
    "tangents.hat_se3": ((6,),),
    "tangents.hat_so3": ((3,),),
    "tangents.left_jacobian_inverse_se3": ((6,),),
    "tangents.left_jacobian_inverse_so3": ((3,),),
    "tangents.left_jacobian_se3": ((6,),),
    "tangents.left_jacobian_so3": ((3,),),
    "tangents.right_jacobian_inverse_se3": ((6,),),
    "tangents.right_jacobian_inverse_so3": ((3,),),
    "tangents.right_jacobian_se3": ((6,),),
    "tangents.right_jacobian_so3": ((3,),),
    "tangents.vee_se3": ((4, 4),),
    "tangents.vee_so3": ((3, 3),),
}
#: Public functions without tensor inputs.
NO_TENSOR_INPUTS = {"so3.identity", "se3.identity"}
MODULES = {"so3": so3, "se3": se3, "tangents": tangents}


def _function(qualified: str):
    module, name = qualified.split(".")
    return getattr(MODULES[module], name)


def test_the_table_covers_every_public_function() -> None:
    public = {f"{module}.{name}" for module, namespace in MODULES.items() for name in namespace.__all__}
    assert public == set(PARAMETERS) | NO_TENSOR_INPUTS


def _check(qualified: str) -> None:
    function, shapes = _function(qualified), PARAMETERS[qualified]
    check_boundary(function, dict(zip(inspect.signature(function).parameters, shapes)))


def test_public_functions_reject_bad_inputs() -> None:
    failed = failures(_check, sorted(PARAMETERS))
    assert not failed, "\n".join(failed)
