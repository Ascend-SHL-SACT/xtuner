from xtuner.v1.config import (
    AdamWConfig,
    MuonConfig,
    LRConfig,
)
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig, PretrainTokenizeFunctionConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE397BA17SplitConfig
from xtuner.v1.model.moe.moe import MTPConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os
import shutil
from pathlib import Path

# ===== A3 NPU 实验配置 =====
# 数据集: 10 个 (纯多模态, 含图像/视频)
# 规模: 8 节点 × 16 Die = 128 Die, sp_size=4, dp_size=32, gbs=32

ceph_config = "/llmit-data/yujiashuo/petreloss.conf"
# meta_data_path = "/llmit-data/yujiashuo/a3_base05/metas/interns2_pre_base05_20260424a_a3_test10_mm.json"
meta_data_path = "/llmit-data/yujiashuo/a3_base05/metas/interns2_pre_base05_20260424a_a3.json"
# meta_data_path = "/mnt/huawei/hyf/xtuner_0410/hyf_test/test_yidian.json"
model_path = "/mnt/huawei/weight/Qwen3.5-397B-A17B-split"
work_dir = "/mnt/huawei/hyf/xtuner_logs_0410_397b"
tokenizer_cache_dir = "/llmit-data/yujiashuo/a3_base05/tokenizer_cache"

if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
shutil.copy(__file__, work_dir)

# 训练超参数
sample_max_length = 64 * 1024
pack_max_length = 64 * 1024
rand_video_max_frames = 24
num_workers = 2
global_batch_size = 128          # dp_size = 128/sp_size(4) = 32
total_epoch = 1
hf_interval = 10
hf_max_keep = 2
checkpoint_interval = 10
checkpoint_maxkeep = 1
lr = 2e-5
lr_min = 1e-6
weight_decay = 0.05
warmup_ratio = 0.1
recompute_ratio = 1.0
loss_reduction = "square"
max_pixels = 16777216
sp_size = 4
ep_size = 16

# model config
model_cfg = Qwen3_5_VLMoE397BA17SplitConfig()
# model_cfg.text_config.num_hidden_layers = 20
model_cfg.text_config.router_async_offload = True
model_cfg.text_config.ep_size = ep_size

with (Path(model_path) / "config.json").open() as f:
    model_hf_config = json.load(f)
model_cfg.text_config.vocab_size = model_hf_config["text_config"]["vocab_size"]

NORMAL_MTP_LAYERS = 4
NORMAL_MTP_FACTOR = 1.0
model_cfg.text_config.mtp_config = MTPConfig(num_layers=NORMAL_MTP_LAYERS, share_weights=NORMAL_MTP_LAYERS > 1, loss_scaling_factor=NORMAL_MTP_FACTOR)


# dataset config
if ceph_config is not None:
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
else:
    oss_loader_cfg = None

has_pretrain = False
ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    if _data.get("text_pretrain", False):
        has_pretrain = True

    class_name = "JsonlDataset" if _data.get("text_pretrain", False) else "VLMJsonlDataset"

    if _data.get("text_pretrain", False):
        tokenize_fn = PretrainTokenizeFunctionConfig(hash=_data.get("hash", None))
    else:
        tokenize_fn = Qwen3VLTokenizeFnConfig(
            chat_template="qwen3.5-vl",
            llm_pack_weight=-3.2,
            visual_pack_weight=5.0,
            max_length=sample_max_length,
            processor_path=model_path,
            rand_video_max_frames=rand_video_max_frames,
            oss_loader_cfg=oss_loader_cfg,
            max_pixels=max_pixels,
            debug=True,
        )

    _data_cfg = {
        "dataset": DatasetConfig(
            name=name,
            anno_path=_data["annotation"],
            media_root=_data.get("media_root", ""),
            sample_ratio=_data.get("sample_ratio", 1.0),
            class_name=class_name,
            enable_sequential_sampler=True,
            cache_tag="cache_tags_v1",
            cache_dir=tokenizer_cache_dir,
        ),
        "tokenize_fn": tokenize_fn,
    }
    dataset_config.append(_data_cfg)

if has_pretrain:
    pack_level = "mllm_hybrid"
else:
    pack_level = "soft"

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_level=pack_level,
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
)
# optim_cfg = AdamWConfig(lr=6e-05, foreach=False)
optim_cfg = MuonConfig(lr=lr, weight_decay=weight_decay)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(
    recompute_ratio=recompute_ratio,
    torch_compile=True,
    ep_size=ep_size,
    checkpoint_preserve_rng_state=False,
)

# resume_cfg = ResumeConfig(auto_resume=True)

trainer = TrainerConfig(
    sp_size=sp_size,
    load_from=model_path,
    # resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    # hf_interval=hf_interval,
    # checkpoint_interval=checkpoint_interval,
    # checkpoint_maxkeep=checkpoint_maxkeep,
    # hf_max_keep=hf_max_keep,
    work_dir=work_dir,
    profile_step=5,
    profile_time=True,
    profile_memory=True,
    seed=0,
    strict_load=False,
)

from xtuner.v1.patch.fully_shard_patch import apply_fully_shard_patch
apply_fully_shard_patch()
