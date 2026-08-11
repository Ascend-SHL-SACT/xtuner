import os

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import OpenaiTokenizeFunctionConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import get_model_config_from_hf
from xtuner.v1.train import TrainerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig


def _get_bool_env(name, default=False):
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


# Model config from slim model (4 layers, 128 experts)
GLM5_2_MODEL_PATH = "/mnt/intern-delivery-shared/jschen/models/glm52-slim"
ALPACA_PATH = "/mnt/intern-delivery-shared/jschen/datasets/tatsu-lab-alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet"

work_dir = os.environ.get("WORK_DIR", "/weight/jschen/work_dirs/glm52_pretrain")
ep_size = int(os.environ.get("EP_SIZE", "8"))
sample_max_length = int(os.environ.get("SAMPLE_MAX_LENGTH", "4096"))
pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", "16384"))
total_step = int(os.environ.get("TOTAL_STEP", "10"))
global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "16"))

loss_cfg = CELossConfig(
    mode=os.environ.get("LOSS_MODE", "chunk"),
    chunk_size=int(os.environ.get("LOSS_CHUNK_SIZE", "1024")),
)

# Build model config from HF config (slim model)
model_cfg = get_model_config_from_hf(GLM5_2_MODEL_PATH)
model_cfg.dispatcher = None  # single node, no dispatcher
model_cfg.ep_size = ep_size
model_cfg.compile_cfg = False  # disable torch.compile on NPU
model_cfg.float8_cfg = None  # disable float8 on NPU
model_cfg.lm_loss_cfg = loss_cfg

# Auto-select NPU sparse MLA backend
if hasattr(model_cfg.attention, "sparse_mla_backend"):
    model_cfg.attention.sparse_mla_backend = os.environ.get("SPARSE_MLA_BACKEND", "npu")

cache_dir = os.path.join(work_dir, "jsonl_cache")
dataset_config = [
    {
        "dataset": DatasetConfig(
            name="alpaca",
            anno_path=ALPACA_PATH,
            sample_ratio=1.0,
            cache_dir=cache_dir,
            cache_tag=f"glm52_{sample_max_length}",
        ),
        "tokenize_fn": OpenaiTokenizeFunctionConfig(
            chat_template="glm5.2",
            max_length=sample_max_length,
        ),
    }
]

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_level=os.environ.get("PACK_LEVEL", "soft"),
    pack_max_length=pack_max_length,
    pack_chunk_size=int(os.environ.get("PACK_CHUNK_SIZE", "10000")),
    pack_workers=int(os.environ.get("PACK_WORKERS", "4")),
    global_pack=_get_bool_env("GLOBAL_PACK", True),
    group_by_length=_get_bool_env("GROUP_BY_LENGTH", True),
    num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "4")),
)

optim_cfg = AdamWConfig(
    lr=float(os.environ.get("LR", "1e-4")),
    foreach=False,
    swap_optimizer=False,
)
lr_cfg = LRConfig(lr_type=os.environ.get("LR_TYPE", "cosine"), warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.1")))

fsdp_cfg = FSDPConfig(
    cpu_offload=False,
    ep_size=ep_size,
    torch_compile=False,
)

trainer = TrainerConfig(
    model_cfg=model_cfg,
    load_from=None,  # Train from scratch
    tokenizer_path=GLM5_2_MODEL_PATH,
    strict_load=False,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=loss_cfg,
    fsdp_cfg=fsdp_cfg,
    global_batch_size=global_batch_size,
    total_step=total_step,
    intra_layer_micro_batch=int(os.environ.get("INTRA_LAYER_MICRO_BATCH", "1")),
    sp_size=int(os.environ.get("SP_SIZE", "1")),
    load_checkpoint_cfg=LoadCheckpointConfig(checkpoint_path=os.environ.get("LOAD_CHECKPOINT_PATH")),
    checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "200")),
    checkpoint_maxkeep=int(os.environ.get("CHECKPOINT_MAX_KEEP", "3")),
    hf_interval=int(os.environ.get("HF_INTERVAL", "200")),
    hf_max_keep=int(os.environ.get("HF_MAX_KEEP", "3")),
    work_dir=work_dir,
    profile_memory=_get_bool_env("PROFILE_MEMORY", False),
    profile_time=_get_bool_env("PROFILE_TIME", False),
    profile_step=[int(x) for x in os.environ.get("PROFILE_STEP", "2,3").split(",") if x],
    debug_skip_save=_get_bool_env("DEBUG_SKIP_SAVE", False),
    dist_backend="npu:hccl",
)
