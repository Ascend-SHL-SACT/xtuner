# cp /mnt/hw-linyifei2/pkgs/comm_repair_0206_1/libhccl.so /usr/local/Ascend/ascend-toolkit/8.3.RC3/lib64
# cp /mnt/hw-linyifei2/pkgs/comm_repair_0206_1/libhccl_plf.so /usr/local/Ascend/ascend-toolkit/8.3.RC3/lib64
# cp /mnt/hw-linyifei2/pkgs/comm_repair_0206_1/Ascend-aicpu_extend_syskernels.tar.gz /usr/local/Ascend/ascend-toolkit/8.3.RC3/opp/Ascend/aicpu
# cp /mnt/hw-linyifei2/pkgs/comm_repair_0206_1/Ascend-aicpu_syskernels.tar.gz /usr/local/Ascend/ascend-toolkit/8.3.RC3/opp/Ascend/aicpu

# pip install astor
# pip install /mnt/hw-linyifei2/pkgs/pta_0212_force_affinity/torch_npu-2.7.1.dev20260212-cp311-cp311-manylinux_2_28_aarch64.whl

# pip install transformers==5.3.0
# pip install triton_ascend==3.2.0

config_file=${1}
datetime=$(date +%Y%m%d_%H%M%S)
log_dir="logs/${datetime}"

export HCCL_BUFFSIZE=16
export XTUNER_TOKENIZE_WORKERS=1

export HCCL_CONNECT_TIMEOUT=600
export HCCL_EXEC_TIMEOUT=600

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,segment_size_mb:128
export MOE_AIV=1
export XTUNER_ACTIVATION_OFFLOAD=1
export CPU_AFFINITY_FORCE=True
export CPU_AFFINITY_CONF=1,npu0:12-23,npu1:26-37,npu2:52-63,npu3:66-77,npu4:92-103,npu5:106-117,npu6:132-143,npu7:146-157,npu8:172-183,npu9:186-197,npu10:212-223,npu11:226-237,npu12:252-263,npu13:266-277,npu14:292-303,npu15:306-317
# unset XTUNER_ACTIVATION_OFFLOAD

export TASK_QUEUE_ENABLE=1
export PYTHONPATH=/mnt/huawei/wtg/xtuner_hyf_qwen35:$PYTHONPATH
export PYTHONPATH=/mnt/hwfile/vc-intern-delivery/yehaochen/codebase/interntrain:$PYTHONPATH 
export PYTHONPATH=/mnt/hwfile/vc-intern-delivery/vl_delivery/code/mmengine:$PYTHONPATH

# 兼容clusterx训练和单机训练
export XTUNER_USE_FA3=${XTUNER_USE_FA3-"1"}
export TORCH_LOGS=${TORCH_LOGS-"recompiles"}

# /mnt/huawei/hyf/CANN/8.5.0.B030/Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install
# /mnt/huawei/hyf/CANN/8.5.0.B030/Atlas-A3-cann-kernels_8.5.0_linux-aarch64.run --install

NNODES=$WORLD_SIZE  # WORLD_SIZE是clusterx训练时注入的环境变量
if [ "x${NNODES}" == "x" ]; then
NNODES=${NODE_COUNT-"1"}  # NODE_COUNT等是仪电训练任务调度器注入的环境变量
fi
NRANK=${RANK}  # clusterx
if [ "x${NRANK}" == "x" ]; then
NRANK=${NODE_RANK-"0"}  # yidian
fi
NPROC_PER_NODE=${GPUS_PER_NODE}  # clusterx
if [ "x${NPROC_PER_NODE}" == "x" ]; then 
NPROC_PER_NODE=${PROC_PER_NODE-"8"}  # yidian
fi

# hyf
export MULTI_STREAM_MEMORY_REUSE=2
export MODEL_PATH=/mnt/huawei/weight/Qwen3.5-35B-A3B
export MEDIA_ROOT=''
export DATA_PATH='/mnt/hwfile/vc-intern-delivery/vl_delivery/code/huawei_debug/task_entries/meta_data/export_meta_internvl3_5_internvlm3_tiny_final_debug.json'
export WORK_DIR="/mnt/huawei/hyf/"
NRANK=0
NNODES=1
NPROC_PER_NODE=8
MASTER_ADDR=${MASTER_ADDR-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT-"6002"}
DISTRIBUTED_ARGS="--nproc_per_node $NPROC_PER_NODE --nnodes $NNODES --node_rank $NRANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

run_cmd="torchrun $DISTRIBUTED_ARGS -m xtuner.v1.train.cli.sft --config ${config_file}"

env

# 支持日志同时输出到文件，可以替代仪电页面上查看日志。这通过 log_dir 是否传入决定
if [ "x${log_dir}" == "x" ]; then
    # console日志不输出到文件
    echo "run_cmd: ${run_cmd}" 
    eval ${run_cmd}

else
    # console日志输出到文件
    mkdir -p ${log_dir}
    env | tee -a "${log_dir}/node_${NRANK}.txt"
    echo "---------------------------START--------------------------------" | tee -a "${log_dir}/node_${NRANK}.txt"
    echo "run_cmd: ${run_cmd}" | tee -a "${log_dir}/node_${NRANK}.txt"
    set -o pipefail
    eval ${run_cmd} 2>&1 | tee -a "${log_dir}/node_${NRANK}.txt"
    status=$?
    if [ $status -ne 0 ]; then exit $status; fi
    echo "---------------------------END--------------------------------" | tee -a "${log_dir}/node_${NRANK}.txt"
fi
