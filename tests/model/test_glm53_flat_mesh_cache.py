"""MoE 平坦 replicate 子网格缓存回归测试（``XTUNER_MOE_FLAT_MESH_CACHE``）。

``xtuner/v1/model/moe/moe.py`` 的 ``scale_and_reduce_grad`` 每个 replicate 梯度都要
``DeviceMesh.__getitem__``/``_flatten`` 重建子网格，新增 ``_FLAT_REPLICATE_MESH_CACHE``
（WeakKeyDictionary，键为父网格）+ ``_flat_replicate_mesh`` 缓存派生子网格；
``XTUNER_MOE_FLAT_MESH_CACHE=0`` 恢复逐次重建。

用例全部单进程 CPU gloo（world_size=1 的 FileStore init），构造 1x1 的 ("dp","tp") 网格即可
验证子网格等价性、缓存命中恒等、weak-key 随父网格回收、以及 ``scale_and_reduce_grad`` 的
flag 路由与返回值被真实用于 grad 缩放/all_reduce 分组。
"""

import gc
import os
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate

import xtuner.v1.model.moe.moe as moe


_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_ENV_PROBE_CODE = "import xtuner.v1.model.moe.moe as m; print(m._MOE_FLAT_MESH_CACHE)"


@pytest.fixture()
def gloo_world1(tmp_path: Path):
    """单进程 gloo world_size=1：DeviceMesh/DTensor 构造的前置条件。"""
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            pytest.skip("requires a single-process world")
        yield
        return
    dist.init_process_group("gloo", store=dist.FileStore(str(tmp_path / "gloo_store"), 1), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()
        # The world PG is gone: drop any derived-mesh entries pinned to it so a
        # selectively-run / crashed test cannot leak dead-group entries into later
        # files of the same pytest session.
        moe._FLAT_REPLICATE_MESH_CACHE.clear()


def _mesh_1x1() -> DeviceMesh:
    return init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))


def _mesh_1d() -> DeviceMesh:
    # 1-D mesh for the single-Replicate-placement DTensors below: a fully legal
    # DTensorSpec, unlike one placement on a 2-D mesh (which only works because
    # from_local does not validate placements length today).
    return init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))


class TestFlatReplicateMesh:
    """``_flat_replicate_mesh`` 等价性与缓存命中。"""

    def test_single_replicate_dim_matches_direct_submesh(self, gloo_world1) -> None:
        mesh = _mesh_1x1()
        direct = mesh["dp"]
        moe._FLAT_REPLICATE_MESH_CACHE.clear()
        flat = moe._flat_replicate_mesh(mesh, ("dp",))
        assert flat.size() == direct.size() == 1
        assert flat.ndim == 1
        assert dist.get_process_group_ranks(flat.get_group()) == dist.get_process_group_ranks(direct.get_group())
        # 缓存命中：同 (mesh, names) 二次调用返回同一对象。
        assert moe._flat_replicate_mesh(mesh, ("dp",)) is flat

    def test_multi_replicate_dim_matches_flatten(self, gloo_world1) -> None:
        mesh = _mesh_1x1()
        direct = mesh[("dp", "tp")]._flatten()
        moe._FLAT_REPLICATE_MESH_CACHE.clear()
        flat = moe._flat_replicate_mesh(mesh, ("dp", "tp"))
        assert flat.ndim == 1
        assert flat.size() == direct.size() == 1
        assert torch.equal(flat.mesh, direct.mesh)
        assert dist.get_process_group_ranks(flat.get_group()) == dist.get_process_group_ranks(direct.get_group())
        assert moe._flat_replicate_mesh(mesh, ("dp", "tp")) is flat

    def test_value_equal_parents_share_entry(self, gloo_world1) -> None:
        # torch 的 DeviceMesh 按值比较/哈希：两次 init_device_mesh 同参构造出两个不同对象但
        # hash/eq 相同，WeakKeyDictionary 因此视作同一键。两条目共享的平坦网格对两个父网格都
        # 与直接构造等值，语义无害，这里把该行为钉住。
        mesh_a = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
        mesh_b = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
        assert mesh_a is not mesh_b
        moe._FLAT_REPLICATE_MESH_CACHE.clear()
        flat_a = moe._flat_replicate_mesh(mesh_a, ("dp",))
        flat_b = moe._flat_replicate_mesh(mesh_b, ("dp",))
        assert flat_a is flat_b, "值相等（hash/eq 相同）的父网格共享同一条目"
        for mesh in (mesh_a, mesh_b):
            direct = mesh["dp"]
            assert flat_a.size() == direct.size()
            assert dist.get_process_group_ranks(flat_a.get_group()) == dist.get_process_group_ranks(direct.get_group())
        assert mesh_a in moe._FLAT_REPLICATE_MESH_CACHE

    def test_weakkey_cache_does_not_pin_parent_mesh(self, gloo_world1) -> None:
        # 缓存容器自身是 weak-key 的：条目不额外持父网格强引用。注意（torch 2.9.1 观测）：
        # 一旦派生过子网格（mesh[...]/_flatten），torch 的 mesh 资源注册表会强引用父网格，
        # 父网格的回收时机由 torch 侧决定，与本缓存的 weak-key 语义无关；因此这里用未派生
        # 子网格的父网格验证容器语义。
        moe._FLAT_REPLICATE_MESH_CACHE.clear()
        mesh = _mesh_1x1()
        moe._FLAT_REPLICATE_MESH_CACHE[mesh] = {}
        assert mesh in moe._FLAT_REPLICATE_MESH_CACHE
        ref = weakref.ref(mesh)
        del mesh
        gc.collect()
        assert ref() is None, "缓存容器不得阻止父网格回收"
        assert len(moe._FLAT_REPLICATE_MESH_CACHE) == 0, "WeakKeyDictionary 条目应随键消亡"


class TestScaleAndReduceGradFlagRouting:
    """``scale_and_reduce_grad`` 的 flag 路由与平坦网格的实际使用。"""

    def _fake_model(self, dtensor: DTensor) -> SimpleNamespace:
        return SimpleNamespace(
            trainable_parameters=lambda: [("layers.0.mlp.down_proj", dtensor)],
            ep_mesh=None,
            config=SimpleNamespace(expert_tp_size=1),
        )

    def _dtensor_with_grad(self, mesh: DeviceMesh) -> DTensor:
        local = torch.randn(2, 2, dtype=torch.float32, requires_grad=True)
        dt = DTensor.from_local(local, mesh, placements=[Replicate()])
        dt.grad = torch.ones(2, 2, dtype=torch.float32) * 3.0
        return dt

    def test_flag_on_uses_helper_result_for_scaling(self, gloo_world1, monkeypatch: pytest.MonkeyPatch) -> None:
        mesh = _mesh_1d()
        dt = self._dtensor_with_grad(mesh)
        calls: list[tuple[DeviceMesh, tuple[str, ...]]] = []
        # 假 helper 返回 size=4 的平坦网格：grad 必须被除以 4，且 all_reduce 走其 group。
        fake_flat = SimpleNamespace(size=lambda: 4, get_group=lambda: dist.group.WORLD)

        def fake_helper(parent_mesh: DeviceMesh, names: tuple[str, ...]) -> object:
            calls.append((parent_mesh, names))
            return fake_flat

        monkeypatch.setattr(moe, "_MOE_FLAT_MESH_CACHE", True)
        monkeypatch.setattr(moe, "_flat_replicate_mesh", fake_helper)
        moe.MoE.scale_and_reduce_grad(self._fake_model(dt))
        assert calls == [(mesh, ("dp",))], "flag 开必须经 _flat_replicate_mesh 取平坦网格"
        assert torch.equal(dt.grad, torch.full((2, 2), 0.75)), "grad 应按平坦网格 size 原地缩放"

    def test_flag_off_rebuilds_without_helper(self, gloo_world1, monkeypatch: pytest.MonkeyPatch) -> None:
        mesh = _mesh_1d()
        dt = self._dtensor_with_grad(mesh)
        calls: list[object] = []

        def spy(*args: object, **kwargs: object) -> object:
            calls.append(1)
            raise AssertionError("flag 关不应调用 _flat_replicate_mesh")

        monkeypatch.setattr(moe, "_MOE_FLAT_MESH_CACHE", False)
        monkeypatch.setattr(moe, "_flat_replicate_mesh", spy)
        moe.MoE.scale_and_reduce_grad(self._fake_model(dt))
        assert calls == []
        # 1x1 网格的平坦 size 为 1：grad 不被缩放（除以 1），all_reduce 本地无聚合效果。
        assert torch.equal(dt.grad, torch.full((2, 2), 3.0))


class TestFlatMeshCacheFlagDefault:
    """``XTUNER_MOE_FLAT_MESH_CACHE`` 默认值契约（干净子进程 import 探测）。"""

    def _flag_in_subprocess(self, env_value: str | None) -> bool:
        env = dict(os.environ)
        env.pop("XTUNER_MOE_FLAT_MESH_CACHE", None)
        if env_value is not None:
            env["XTUNER_MOE_FLAT_MESH_CACHE"] = env_value
        env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", _ENV_PROBE_CODE],
            env=env,
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=300,
        )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        return lines[-1].strip() == "True"

    def test_flag_defaults_on_and_env_zero_disables(self) -> None:
        assert self._flag_in_subprocess(None) is True
        assert self._flag_in_subprocess("0") is False
