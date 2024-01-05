from pathlib import Path
import json
from collections import OrderedDict
import torch.nn as nn
import torch

def read_json(cfg_fname):
    fname = Path(cfg_fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook = OrderedDict)
    
def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
        
def prepare_device(n_gpu_use):
    """GPU 사용이 가능하면 셋팅하는 메서드. DataParallel에 사용할 gpu 번호를 받는다.

    Args:
        n_gpu_use (_type_): 사용할 GPU 갯수

    Returns:
        device, list_ids (_type_): 사용할 장치(cpu, gpu), 사용할 GPU 번호
    """    
    n_gpu = torch.cuda.device_count()
    if(n_gpu_use > 0 and n_gpu == 0): # 사용한 GPU가 없으면
        print("현재 GPU를 사용할 수 없습니다.")
        n_gpu_use = 0
        
    if(n_gpu_use > n_gpu):
        print("현재 할당가능한 GPU 갯수가 %d 보다 적습니다. \
            가용 가능한 개수만으로 학습을 돌립니다." % n_gpu_use)
        
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0