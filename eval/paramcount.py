import torch
import torch.nn as nn
import importlib.util
import sys

def format_num(num):
    for unit in ['','K','M','B','T']:
        if num < 1000:
            return f"{num:.2f}{unit}"
        num /= 1000
    return f"{num:.2f}P"

if __name__ == "__main__":
    # 动态导入模型文件
    model_path = "/mnt/dataset1/ldy/Workspace/FLORA/model/EEG_MedformerNoTS.py"
    spec = importlib.util.spec_from_file_location("EEG_MedformerNoTS", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["EEG_MedformerNoTS"] = module
    spec.loader.exec_module(module)
    
    # 创建模型实例
    model = module.eeg_encoder()
    
    # 打印每层的参数状态
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    
    # 计算并打印模型的总参数量和可训练参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")