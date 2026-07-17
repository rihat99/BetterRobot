"""Identity and import contracts for the optimizer-owned ``CostStack``."""

from __future__ import annotations

import better_robot as br
from better_robot.costs import CostItem as PackageShimCostItem
from better_robot.costs import CostStack as PackageShimCostStack
from better_robot.costs.stack import CostItem as ModuleShimCostItem
from better_robot.costs.stack import CostKind as ModuleShimCostKind
from better_robot.costs.stack import CostStack as ModuleShimCostStack
from better_robot.optim.cost_stack import CostItem as CanonicalCostItem
from better_robot.optim.cost_stack import CostKind as CanonicalCostKind
from better_robot.optim.cost_stack import CostStack as CanonicalCostStack


def test_cost_stack_has_one_canonical_class_identity() -> None:
    assert PackageShimCostStack is CanonicalCostStack
    assert ModuleShimCostStack is CanonicalCostStack
    assert br.CostStack is CanonicalCostStack
    assert CanonicalCostStack.__module__ == "better_robot.optim.cost_stack"


def test_cost_item_and_kind_alias_share_the_canonical_identity() -> None:
    assert PackageShimCostItem is CanonicalCostItem
    assert ModuleShimCostItem is CanonicalCostItem
    assert CanonicalCostItem.__module__ == "better_robot.optim.cost_stack"
    assert ModuleShimCostKind is CanonicalCostKind


def test_cost_stack_public_import_is_constructible() -> None:
    stack = CanonicalCostStack()
    assert stack.items == {}
