import torch
import torch_npu
import time
import os
import csv
from datetime import datetime


class CUDAEVENT_TIMER():
    def __init__(self):
        self.events = {}
        self.times = {}
        self.cpu_times = []
        self.shapes = {}
        self.flush_count = 0  # 记录 flush 次数
    
    def add_cpu(self):
        torch.npu.synchronize()
        self.cpu_times.append(time.time())

    def add_tensor_shape(self, label, tensor):
        assert isinstance(tensor, torch.Tensor), f"tensor {tensor} is not a torch.Tensor"
        if label not in self.shapes:
            self.shapes[label] = tensor.shape
    
    def add_tensor(self, label, tensor):
        if label not in self.shapes:
            self.shapes[label] = tensor

    def add_event(self, label, group=None):
        if label not in self.times:
            self.times[label] = []
            self.events[label] = []
        stream = None
        if group is not None:
            collective_stream_id = group._get_backend(torch.device('npu'))._get_stream_id(False)
            if collective_stream_id is None:
                return
            stream = torch_npu.npu.Stream(stream_id=collective_stream_id, device_type=20)

        event = torch_npu.npu.Event(enable_timing=True)
        event.record(stream=stream)
        self.events[label].append(event)

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
        
        # 收集本次 flush 的数据
        row_data = {
            'timestamp': datetime.now().isoformat(),
            'flush_count': self.flush_count,
            'device': torch.npu.current_device(),
        }
        
        # 处理 NPU event 时间
        for label in self.times:
            events = self.events[label]
            if len(events) % 2 != 0:
                print(f"[rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] got {len(events)} {label} events, should be paired")
                row_data[f'{label}_status'] = 'unpaired'
            else:
                times = []
                for i in range(0, len(events), 2): 
                    t = events[i].elapsed_time(events[i+1])
                    times.append(t)
                avg_time = sum(times) / len(times) if times else 0
                row_data[f'{label}_count'] = len(times)
                row_data[f'{label}_avg_ms'] = avg_time
                row_data[f'{label}_times_ms'] = str(times)
                print(f"\n [rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}] {label} execute {len(times)} times, average time is {avg_time} ms, each time is: {times}")
        
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
