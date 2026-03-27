from xtuner.v1.config import (
    AdamWConfig,
    LRConfig,
    MuonConfig,
)
from xtuner.v1.module.rope.rope import RopeScalingConfig
from xtuner.v1.train import TrainerConfig, ResumeConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig, PretrainTokenizeFunctionConfig
from xtuner.v1.model import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.model.compose.intern_s1 import InternS1VisionConfig, InternS1Config
from xtuner.v1.datasets import InternS1VLTokenizeFnConfig
from xtuner.v1.module.mtp.config import MTPConfig

from xtuner.v1.model.moe.qwen3 import Qwen3MoE235BA22Config
from xtuner.v1.loss import CELossConfig
from xtuner.v1.datasets.config import DatasetConfig, DataloaderConfig
from xtuner.v1.config import FSDPConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
import json
import os
import shutil
from pathlib import Path

# 路径配置
ceph_config = "/mnt/hwfile/vc-intern-delivery/vl_delivery/code/huawei_debug/task_entries/ceph_config/petreloss.conf"
meta_data_path = '/mnt/hwfile/vc-intern-delivery/vl_delivery/code/huawei_debug/task_entries/meta_data/export_meta_internvl3_5_internvlm3_tiny_final_debug.json'
# model_path还需要修改
model_path = '/mnt/huawei/weight/Qwen3.5-35B-A3B'
work_dir = "/mnt/huawei/hyf/"
tokenizer_cache_dir = "/mnt/hwfile/vc-intern-delivery/vl_delivery/cache/xtuner_tokenizer_cache/internvl3.5_tiny/slow_tokenize_sft_ml_32k_tokenizer"


# 将当前配置文件拷贝到work_dir
if not os.path.exists(work_dir):
    os.makedirs(work_dir, exist_ok=True)
current_file = __file__
shutil.copy(current_file, work_dir)

# 训练超参数
sample_max_length = 1 * 1024
pack_max_length = 1 * 1024
intra_layer_micro_batch = 1
processor_path = model_path
rand_video_max_frames = 24
add_vision_id = True
num_workers = 0
global_batch_size = None
total_epoch = 10
hf_interval = 300
hf_max_keep = 50
checkpoint_interval = 2
checkpoint_maxkeep = 50
lr = 2e-5
lr_min = 2e-6
weight_decay = 0.05
warmup_ratio = 0.03
recompute_ratio = 1.0
loss_reduction = "square"
# 这个参数控制是否会生成position_id,xtuner/v1/datasets/mllm_tokenize_fn/qwen3_vl_tokenize_fn.py:483
enable_3d_rope = True
max_pixels = 16777216   # 16384 * 32 * 32

# model config
model_cfg = Qwen3_5_VLMoE35BA3Config(text_config=Qwen3_5_VLTextMoE35BA3BConfig(
                                        # mtp_config=MTPConfig(
                                        #     num_layers=1, 
                                        #     loss_scaling_factor=0.1)
                                        ),
                                    freeze_vision=True, 
                                    freeze_projector=True)
# model_cfg.vision_config.depth = 24
# model_cfg.vision_config.hidden_size = 1024
# model_cfg.vision_config.intermediate_size = 4096
# model_cfg.vision_config.deepstack_visual_indexes = []

# model_cfg.projector_config.vision_hidden_size = 1024
# model_cfg.projector_config.deepstack_visual_indexes = []

# model_cfg.text_config.rope_scaling_cfg = RopeScalingConfig(
#             fope_init_factor=0.5,
#             fope_sep_head=True,
#             num_inv_freq=None,
#             )

# model_cfg.text_config.vocab_size = 155008

# model_cfg.text_config.router.use_grouped_router = True
# model_cfg.text_config.router.router_n_groups = 8

model_cfg.text_config.ep_size = 1
model_cfg.text_config.dispatcher='all2all'

# from xtuner.v1.float8.config import Float8Config, ScalingGranularity

# float8_cfg = Float8Config(
#     scaling_granularity_gemm=ScalingGranularity.TILEWISE,
#     scaling_granularity_grouped_gemm=ScalingGranularity.TILEWISE,
# )
# model_cfg.text_config.float8_cfg = float8_cfg

# dataset config
min_num_frames = 8
max_num_frame = 36
if ceph_config is not None:
    oss_loader_cfg = OSSLoaderConfig(backend_kwargs={"conf_path": ceph_config})
else:
    oss_loader_cfg = None

ds_collections = json.loads(open(meta_data_path).read())
dataset_config = []
for name, _data in ds_collections.items():
    _data_cfg = {"dataset": DatasetConfig(name=name,
                                          anno_path=_data['annotation'],
                                          media_root=_data.get('media_root', ''),
                                          sample_ratio=_data.get('sample_ratio', 1.0),
                                          enable_sequential_sampler=True,
                                          class_name='VLMJsonlDataset',
                                          cache_tag='cache_tags_v1',
                                          cache_dir=tokenizer_cache_dir),
                 "tokenize_fn": Qwen3VLTokenizeFnConfig(
                                        max_length=sample_max_length,
                                        processor_path=processor_path,
                                        min_pixels=_data.get('min_pixels', None),
                                        max_pixels=max_pixels,
                                        video_min_total_pixels=_data.get('video_min_total_pixels', None),
                                        video_max_total_pixels=_data.get('video_max_total_pixels', None),
                                        video_min_frames=_data.get('video_min_frames', None),
                                        video_max_frames=_data.get('video_max_frames', None),
                                        fps=_data.get('fps', None),
                                        rand_video_max_frames=rand_video_max_frames,
                                        add_vision_id=add_vision_id,
                                        system_message=_data.get('system_message', None),
                                        hash=_data.get('hash', None),
                                        enable_3d_rope=enable_3d_rope,
                                        oss_loader_cfg=oss_loader_cfg,
                                        debug=False,
                                        oss_time_log_thr=10
                                    )
                 }
    dataset_config.append(_data_cfg)
    break

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    pack_to_max_length=True,
    collator="qwen3_vl_sft_collator",
    num_workers=num_workers,
    pack_extra_buffer_size=20,
    # pack_level='none'
)

# optimizer and lr config
# optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=True, swap_optimizer=False)
optim_cfg = AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=True, )
# optim_cfg = MuonConfig(lr=lr, weight_decay=weight_decay)
lr_cfg = LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min)
fsdp_cfg = FSDPConfig(recompute_ratio=recompute_ratio,
                      ep_size=1,
                      # It seems like there is no reference to `tor` in the provided code snippet. If
                      # you could provide more context or clarify where `tor` is mentioned, I would be
                      # happy to help you understand its purpose or functionality.
                      torch_compile=True,
                      checkpoint_preserve_rng_state=False)

resume_cfg = ResumeConfig(auto_resume=False)

# trainer config
trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=resume_cfg,
    tokenizer_path=model_path,
    fsdp_cfg=fsdp_cfg,
    exp_tracker='tensorboard',
    model_cfg=model_cfg,
    optim_cfg=optim_cfg,
    dataloader_cfg=dataloader_config,
    lr_cfg=lr_cfg,
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    # patch_for_dcp_finish=True,
    # total_step=1,
    # total_epoch=total_epoch,
    total_step=10,
    # hf_interval=hf_interval,
    # checkpoint_interval=checkpoint_interval,
    # checkpoint_maxkeep=checkpoint_maxkeep,
    # hf_max_keep=hf_max_keep,
    work_dir=work_dir,
    intra_layer_micro_batch=intra_layer_micro_batch,
    profile_step=5,
    # profile_time=True,
    # profile_memory=False,
    sp_size=1
)
