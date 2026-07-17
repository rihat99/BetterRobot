"""SE3 storage aliases Warp transforms without copies."""

from __future__ import annotations

import os

import pytest
import torch

warp = pytest.importorskip("warp")
warp.config.kernel_cache_dir = os.environ.get(
    "WARP_CACHE_PATH", "/tmp/betterrobot-warp-cache"
)
warp.init()


@pytest.mark.parametrize(
    ("dtype", "warp_dtype"),
    ((torch.float32, warp.transformf), (torch.float64, warp.transformd)),
)
def test_pose_table_pointer_and_stride_alias(dtype, warp_dtype):
    poses = torch.zeros(5, 7, dtype=dtype)
    poses[:, 6] = 1.0
    array = warp.from_torch(poses, dtype=warp_dtype, requires_grad=False)
    assert array.ptr == poses.data_ptr()
    assert tuple(array.strides) == (poses.stride(0) * poses.element_size(),)
