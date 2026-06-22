import torch
import torch_npu
import time
import os
import csv
import functools
from datetime import datetime


class CUDAEVENT_TIMER():
    def __init__(self):
        self.events = {}
        self.times = {}
        self.cpu_times = []
        self.shapes = {}
        self.flush_count = 0
        self.label_to_func = {}
        self.csv_output = False
            
    
    def add_cpu(self):
        torch.npu.synchronize()
        self.cpu_times.append(time.time())

    def add_tensor_shape(self, label, tensor):
        assert isinstance(tensor, torch.Tensor), f"tensor {tensor} is not a torch.Tensor"
        if label not in self.shapes:
            self.shapes[label] = tensor.shape

    def _get_stream_from_group(self, group):
        if group is None:
            return None
        stream_id = group._get_backend(torch.device('npu'))._get_stream_id(False)
        if stream_id is None:
            return None
        elif stream_id < 0:
            return torch.npu.current_stream()
        else:
            return torch_npu.npu.Stream(
                stream_id=stream_id, device_type=20,
                device_index=torch.distributed.get_rank() % 16,
            )

    def add_event(self, label, group=None):
        if label not in self.times:
            self.times[label] = []
            self.events[label] = []
        stream = self._get_stream_from_group(group)
        event = torch_npu.npu.Event(enable_timing=True)
        event.record(stream=stream)
        self.events[label].append(event)

    def patch_fsdp_all_gather(self, label, patch, group=None):
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group
        _orig_foreach_all_gather = _fsdp_collectives.foreach_all_gather
        

        @functools.wraps(_orig_foreach_all_gather)
        @torch.no_grad()
        def _patched_foreach_all_gather(
            fsdp_params, process_group, async_op,
            all_gather_copy_in_stream, all_gather_stream, device,
        ):
            # if torch.distributed.get_rank()==0:
            #     breakpoint()
            # torch.distributed.barrier()
            stream = self._get_stream_from_group(group) if group is not None else all_gather_stream
            ev_start = torch_npu.npu.Event(enable_timing=True)
            ev_end = torch_npu.npu.Event(enable_timing=True)
            ev_start.record(stream=stream)

            result = _orig_foreach_all_gather(
                fsdp_params, process_group, async_op,
                all_gather_copy_in_stream, all_gather_stream, device,
            )

            ev_end.record(stream=stream)
            if label not in self.times:
                self.times[label] = []
                self.events[label] = []
            self.events[label].append(ev_start)
            self.events[label].append(ev_end)
            return result
        if patch:
            _fsdp_collectives.foreach_all_gather = _patched_foreach_all_gather
            _fsdp_param_group.foreach_all_gather = _patched_foreach_all_gather
            self.label_to_func[label] = _orig_foreach_all_gather
        else:
            if label not in self.label_to_func:
                raise ValueError(f"label {label} not patched")
            _fsdp_collectives.foreach_all_gather = self.label_to_func[label]
            _fsdp_param_group.foreach_all_gather = self.label_to_func[label]

    def _get_log_dir(self):
        """获取日志目录，从环境变量 LOG_DIR 获取，默认为 ./logs"""
        return os.environ.get('LOG_DIR', './logs')

    def _get_csv_path(self):
        """获取 CSV 文件路径，每个 rank 一个文件"""
        log_dir = self._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return os.path.join(log_dir, f'event_timer_rank{rank}.csv')

    def flush(self):
        torch.npu.synchronize()
        self.flush_count += 1
        if not self.csv_output:
            # 重置
            self.cpu_times = []
            self.events = {}
            self.times = {}
            self.shapes = {}
            return
        # 收集本次 flush 的数据
        row_data = {
            'timestamp': datetime.now().isoformat(),
            'flush_count': self.flush_count,
            'device': torch.npu.current_device(),
        }
        
        # 处理 NPU event 时间
        for label in sorted(self.times):
            events = self.events[label]
            if len(events) % 2 != 0:
                print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] got {len(events)} {label} events, should be paired")
                row_data[f'{label}_status'] = 'unpaired'
            else:
                times = []
                for i in range(0, len(events), 2): 
                    t = events[i].elapsed_time(events[i+1])
                    times.append(t)
                row_data[f'{label}_max_ms'] = max(times) if times else 0
                row_data[f'{label}_min_ms'] = min(times) if times else 0
                row_data[f'{label}_avg_ms'] = sum(times) / len(times) if times else 0
                row_data[f'{label}_times_ms'] = str(times)
                # print(f"\n [rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] {label} execute {len(times)} times, average time is {avg_time} ms, each time is: {times}")
        
        # 处理 CPU 时间
        if len(self.cpu_times) == 2:
            cpu_time_ms = (self.cpu_times[1] - self.cpu_times[0]) * 1000
            row_data['cpu_time_ms'] = cpu_time_ms
            print(f"\n [rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] cpu_time: {cpu_time_ms} ms")
        
        for label in self.shapes:
            row_data[f'{label}_shape'] = str(self.shapes[label])
        
        # 保存到 CSV 文件
        self._save_to_csv(row_data)
        # 重置
        self.cpu_times = []
        self.events = {}
        self.times = {}
        self.shapes = {}
    
    def _save_to_csv(self, row_data):
        """将数据追加写入 CSV 文件"""
        csv_path = self._get_csv_path()
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writeheader()
            
            # 写入数据行
            writer.writerow(row_data)


event_timer = CUDAEVENT_TIMER()
