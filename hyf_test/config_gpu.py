import os
from xtuner.v1.model.moe.qwen3 import Qwen3MoE30BA3Config
from xtuner.v1.train import TrainerConfig
from xtuner.v1.config import (
    AdamWConfig,
    FSDPConfig,
    LRConfig,
    MuonConfig,
)
from xtuner.v1.datasets import FTDPTokenizeFnConfig
from xtuner.v1.loss.ce_loss import CELossConfig
try:
    from xtuner.v1.loss.moe_loss import ZLossConfig
except:
    from xtuner.v1.model.moe.moe import ZLossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.model.compose.qwen3_5 import Qwen3_5_VLMoE35BA3SplitConfig, Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE397BA17SplitConfig
try:
    from xtuner.v1.model.moe.moe import MTPConfig
except:
    MTPConfig = None

from xtuner.v1.datasets import PretrainTokenizeFunctionConfig
from xtuner.v1.float8.config import Float8Config, ScalingGranularity
# from xtuner.v1.loss.layer_moe_loss import LayerBalancingLossConfig

# QWEN3_MOE_PATH = "/mnt/huawei/weight/Qwen3.5-35B-A3B"
QWEN3_MOE_PATH = "/mnt/huawei/weight/Qwen3.5-397B-A17B-split"
# QWEN3_MOE_PATH = "/mnt/shared-storage-user/llmrazor-share/yehaochen/tmp/asdfasfasdf/20260320160600/hf-2"
# ALPACA_PATH = "/mnt/huawei/wsl/datasets"
ALPACA_PATH = "/mnt/huawei/hyf/xtuner_0410/sample_1k"

float8_cfg = Float8Config(
    scaling_granularity_gemm=ScalingGranularity.TILEWISE,
    scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
)
ep_size=1

# moe_cfg = Qwen3_5_VLMoE35BA3Config()
moe_cfg = Qwen3_5_VLMoE397BA17SplitConfig()
moe_cfg.text_config.num_hidden_layers = 2
moe_cfg.text_config.ep_size = ep_size
NORMAL_MTP_LAYERS = 2 # 示例，按需替换
SCI_MTP_LAYERS = 1
NORMAL_MTP_FACTOR = 1.0
SCI_MTP_FACTOR = 0.3
moe_cfg.text_config.mtp_config = MTPConfig(num_layers=NORMAL_MTP_LAYERS , share_weights=NORMAL_MTP_LAYERS>1, loss_scaling_factor=NORMAL_MTP_FACTOR)
# moe_cfg.text_config.mtp_config = [
#     MTPConfig(name="normal", mask_type=None, num_layers=NORMAL_MTP_LAYERS , share_weights=NORMAL_MTP_LAYERS>1, loss_scaling_factor=NORMAL_MTP_FACTOR),
#     # MTPConfig(name="sci", mask_type="v3", num_layers=SCI_MTP_LAYERS, share_weights=SCI_MTP_LAYERS>1, loss_scaling_factor=SCI_MTP_FACTOR),
# ]
# if MTPConfig is not None:
#     moe_cfg.text_config.mtp_config = MTPConfig(num_layers=1, loss_scaling_factor=1.0)
# moe_cfg.text_config.layer_balancing_loss_cfg = LayerBalancingLossConfig()
# optim_cfg = AdamWConfig(lr=6e-05, foreach=False, swap_optimizer=False)
optim_cfg = MuonConfig(lr=6e-05, weight_decay=0.05)
lr_cfg = LRConfig(lr_type="cosine", lr_min=1e-6)
fsdp_cfg = FSDPConfig(
    torch_compile=False,
    cpu_offload=False,
    ep_size=ep_size
)

dataset_config = [
    {
        "dataset": DatasetConfig(name="alpaca", anno_path=ALPACA_PATH, sample_ratio=1.0),
        # "tokenize_fn": FTDPTokenizeFnConfig(max_length=16*1024),
        "tokenize_fn": PretrainTokenizeFunctionConfig(),
    },
]

dataloader_config = DataloaderConfig(
    pack_max_length=4*1024,
    pack_level="hard"
)

loss_cfg = CELossConfig(mode="chunk",chunk_size=2048)


trainer = TrainerConfig(
    total_step=1000,
    load_from=QWEN3_MOE_PATH,
    model_cfg=moe_cfg,
    optim_cfg=optim_cfg,
    fsdp_cfg=fsdp_cfg,
    dataset_cfg=dataset_config,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    tokenizer_path=QWEN3_MOE_PATH,
    global_batch_size=32,
    # hf_interval=2,
    work_dir="/mnt/huawei/hyf/xtuner_logs_0410",
    # seed=0,
    profile_step=5,
    profile_time=True,
    # profile_memory=True,
    sp_size=1,
    # strict_load=False,
)

def seed_all(seed=42):
    import random 
    import os
    import numpy as np
    import torch
    import torch_npu
    from torch_npu.contrib import transfer_to_npu

    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed) 
    os.environ['HCCL_DETERMINISTIC'] = str(True) 
    os.environ['LCCL_DETERMINISTIC'] = str(1) 
    os.environ['CLOSE_MATMUL_K_SHIFT'] = str(1) 
    os.environ['ATB_MATMUL_SHUFFLE_K_ENABLE'] = str(0) 
    os.environ['ATB_LLM_LCOC_ENABLE'] = str(0) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.use_deterministic_algorithms(True) 
    torch_npu.npu.manual_seed(seed) 
    torch_npu.npu.manual_seed_all(seed)
# seed_all()
from xtuner.v1.patch.fully_shard_patch import apply_fully_shard_patch
apply_fully_shard_patch()