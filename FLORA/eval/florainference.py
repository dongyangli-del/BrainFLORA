import os
import sys
import random
import re
import warnings
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader
from loss import ClipLoss

class ExperimentConfig:
    def __init__(self):
        # Environment Settings
        self.env_config = {
            "WANDB_API_KEY": "KEY",
            "WANDB_MODE": "offline",
            "WANDB_SILENT": "true"
        }
        
        # Model Configuration
        self.model_config = {
            'mode': 'in_subject',  # 'in_subject' or 'joint'
        }
        
        # Device Settings
        self.device_preference = 'cuda:3'
        self.device_type = 'gpu'
        
        # Base Model Paths
        self.base_model_paths = {
            'eeg': '/mnt/dataset1/ldy/Workspace/FLORA/models',
            'meg': '/mnt/dataset1/ldy/Workspace/FLORA/models',
            'fmri': '/mnt/dataset1/ldy/Workspace/FLORA/models'
        }
        
        # MoE Projection Path (for joint mode)
        self.unified_model_path = "/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-27_02-32/60.pth"
        
        # Dataset Paths
        self.data_paths = {
            'eeg': "/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz",
            'meg': "/home/ldy/THINGS-MEG/preprocessed_newsplit",
            'fmri': "/home/ldy/fmri_dataset/Preprocessed"
        }
        
        # Subject Configurations
        self.subjects = {
            'eeg': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
            'meg': ['sub-01', 'sub-02', 'sub-03', 'sub-04'],
            'fmri': ['sub-01', 'sub-02', 'sub-03']
        }
        
        self.eval_modality = 'meg'
        self.modalities = ['eeg', 'meg', 'fmri']
        self.classes = {
            'eeg': 200,
            'meg': 200,
            'fmri': 100
        }
        
        self.output_config = {
            'output_dir': './outputs/contrast',
            'project': "train_pos_img_text_rep",
            'entity': "sustech_rethinkingbci",
            'name': "lr=3e-4_img_pos_pro_eeg"
        }
        
        self.eval_k_values = [2, 4, 10]
    
    def get_test_config(self):
        return {
            'subjects': self.subjects[self.eval_modality],
            'classes': self.classes[self.eval_modality]
        }
    
    def get_device(self):
        return torch.device(self.device_preference if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
    
    def setup_environment(self):
        for key, value in self.env_config.items():
            os.environ[key] = value
        wandb.init(mode="disabled")
        warnings.filterwarnings("ignore")

    def get_model_path(self, subject=None):
        """Get model path based on mode and subject"""
        if self.model_config['mode'] == 'joint':
            return self.unified_model_path
        else:  # in_subject mode
            subject_num = subject.split('-')[1]  # Extract subject number
            subject_dir = os.path.join(
                self.base_model_paths[self.eval_modality],
                'in_subject',
                self.eval_modality.upper(),
                f'sub-{subject_num}'
            )
            time_folder = os.listdir(subject_dir)[0]
            return os.path.join(subject_dir, time_folder, '40.pth')

def get_dataset(sub, config):
    """Get dataset based on modality"""
    if config.eval_modality == 'eeg':
        from data_preparing.eegdatasets import EEGDataset
        return EEGDataset(config.data_paths['eeg'], adap_subject=sub, subjects=[sub], train=False)
    elif config.eval_modality == 'meg':
        from data_preparing.megdatasets_averaged import MEGDataset
        return MEGDataset(config.data_paths['meg'], subjects=[sub], train=False)
    elif config.eval_modality == 'fmri':
        from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
        return fMRIDataset(config.data_paths['fmri'], adap_subject=sub, subjects=[sub], train=False)
    else:
        raise ValueError(f"Unsupported modality: {config.eval_modality}")

def get_features(unified_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    unified_model.eval()
    text_features_all = text_features_all[eval_modality].to(device).float() if isinstance(text_features_all, dict) else text_features_all.to(device).float()
    
    if eval_modality == 'meg':
        img_features_all = (img_features_all[eval_modality][::12] if isinstance(img_features_all, dict) else img_features_all[::12]).to(device).float()
    else:
        img_features_all = (img_features_all[eval_modality] if isinstance(img_features_all, dict) else img_features_all).to(device).float()
    
    total_loss = 0
    correct = 0
    top5_correct_count = 0
    total = 0
    loss_func = ClipLoss() 
    all_labels = set(range(text_features_all.size(0)))
    features_tensor = torch.zeros(0, 0)
    
    with torch.no_grad():
        for batch_idx, (modal, data, labels, text, text_features, img, img_features, _, _, sub_ids) in enumerate(dataloader):
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
            subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
            
            # Handle both joint and in-subject modes
            if hasattr(unified_model, 'mode') and unified_model.mode == 'in_subject':
                neural_features = unified_model(data)
            else:
                neural_features = unified_model(data, subject_ids, modal=eval_modality)
            
            logit_scale = unified_model.logit_scale.float()            
            img_loss = loss_func(neural_features, img_features, logit_scale)
            total_loss += img_loss.item()
            
            for idx, label in enumerate(labels):
                possible_classes = list(all_labels - {label.item()})
                selected_classes = random.sample(possible_classes, k-1) + [label.item()]
                selected_img_features = img_features_all[selected_classes]
                
                logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
                logits_single = logits_img
                
                predicted_label = selected_classes[torch.argmax(logits_single).item()]
                if predicted_label == label.item():
                    correct += 1
                
                if k == test_classes:
                    _, top5_indices = torch.topk(logits_single, 5, largest=True)
                    if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:
                        top5_correct_count += 1
                        
                total += 1

    average_loss = total_loss / (batch_idx + 1)
    accuracy = correct / total    
    top5_acc = top5_correct_count / total    
    return average_loss, accuracy, top5_acc, labels, features_tensor.cpu()

def run_experiments(config, device):
    test_config = config.get_test_config()
    results = {
        'test_accuracies': [],
        'test_accuracies_top5': [],
        'v2_accuracies': [],
        'v4_accuracies': [],
        'v10_accuracies': []
    }
    
    print(f"\nStarting experiments in {config.model_config['mode']} mode")
    
    for sub in test_config['subjects']:
        print(f"\nProcessing subject: {sub}")
        
        # Load appropriate model based on mode
        if config.model_config['mode'] == 'joint':
            from model.unified_encoder_multi_tower import UnifiedEncoder
            model = UnifiedEncoder(config.encoder_paths, device)
        else:  # in_subject mode
            # Import appropriate model class based on modality
            if config.eval_modality == 'meg':
                from contrast_retrieval_MEG import MedformerNoTSW as ModelClass
            elif config.eval_modality == 'eeg':
                from contrast_retrieval import ATMS as ModelClass
            elif config.eval_modality == 'fmri':
                from contrast_retrieval_fMRI import ATMS as ModelClass
            model = ModelClass()
        
        # Load model weights
        model_path = config.get_model_path(subject=sub)
        print(f"Loading model from: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Get dataset and dataloader
        test_dataset = get_dataset(sub, config)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        
        # Collect features
        text_features_test_all = test_dataset.text_features
        img_features_test_all = test_dataset.img_features
        
        if config.model_config['mode'] == 'joint':
            text_features_test_all = {config.eval_modality: text_features_test_all}
            img_features_test_all = {config.eval_modality: img_features_test_all}
        
        # Get evaluation results
        test_loss, test_accuracy, top5_acc, _, _ = get_features(
            model, test_loader, device, text_features_test_all,
            img_features_test_all, k=test_config['classes'],
            eval_modality=config.eval_modality, test_classes=test_config['classes']
        )
        
        results['test_accuracies'].append(test_accuracy)
        results['test_accuracies_top5'].append(top5_acc)
        
        # Evaluate for different k values
        for k in config.eval_k_values:
            _, acc, _, _, _ = get_features(
                model, test_loader, device, text_features_test_all,
                img_features_test_all, k=k,
                eval_modality=config.eval_modality, test_classes=test_config['classes']
            )
            results[f'v{k}_accuracies'].append(acc)
        
        # Print subject results
        print(f"\nResults for subject {sub}:")
        print(f" - Test Loss: {test_loss:.4f}")
        print(f" - Test Accuracy: {test_accuracy:.4f}")
        print(f" - Top5 Accuracy: {top5_acc:.4f}")
        for k in config.eval_k_values:
            print(f" - v{k}_acc Accuracy: {results[f'v{k}_accuracies'][-1]:.4f}")
    
    # Print average results
    print("\nAverage results across all subjects:")
    print(f"Test Accuracy: {np.mean(results['test_accuracies']):.4f} ± {np.std(results['test_accuracies']):.4f}")
    print(f"Test Top5 Accuracy: {np.mean(results['test_accuracies_top5']):.4f} ± {np.std(results['test_accuracies_top5']):.4f}")
    for k in config.eval_k_values:
        print(f"v{k}_acc Accuracy: {np.mean(results[f'v{k}_accuracies']):.4f} ± {np.std(results[f'v{k}_accuracies']):.4f}")

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

def main():
    # Initialize configuration
    config = ExperimentConfig()
    config.setup_environment()
    
    # Get device
    device = config.get_device()
    print(f"Using device: {device}")
    
    # Run experiments
    run_experiments(config, device)

if __name__ == "__main__":
    main()

# import os
# os.chdir('/mnt/dataset0/ldy/Workspace/FLORA')
# import sys
# import random
# import re
# import warnings
# import torch
# import wandb
# import numpy as np
# from torch.utils.data import DataLoader

# # Add correct import paths
# sys.path.append("/mnt/dataset0/ldy/Workspace/FLORA")
# sys.path.insert(0,'/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval')
# sys.path.insert(0,'/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval/model')

# from model.unified_encoder_multi_tower import UnifiedEncoder
# from data_preparing.eegdatasets import EEGDataset
# from loss import ClipLoss

# class ExperimentConfig:
#     def __init__(self):
#         # Environment Settings
#         self.env_config = {
#             "WANDB_API_KEY": "KEY",
#             "WANDB_MODE": "offline",
#             "WANDB_SILENT": "true"
#         }
        
#         # Device Settings
#         self.device_preference = 'cuda:3'
#         self.device_type = 'gpu'
        
#         # Model Paths
#         self.encoder_paths = {
#             'eeg': '/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-06_01-46/150.pth',
#             'meg': '/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-11_14-50/150.pth',
#             'fmri': '/mnt/dataset1/ldy/Workspace/EEG_Image_decode/Retrieval/models/contrast/across/ATMS/01-18_01-35/150.pth'
#         }
#         # MoE Projection Path
#         self.unified_model_path = "/mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-27_02-32/60.pth"
#         # /mnt/dataset1/ldy/Workspace/FLORA/models/contrast/across/Unified_EEG+MEG+fMRI_EEG/01-27_02-32/60.pth
#         # Dataset Paths
#         self.data_paths = {
#             'eeg': "/mnt/dataset0/ldy/datasets/THINGS_EEG/Preprocessed_data_250Hz",  # Updated to match working code
#             'meg': "/home/ldy/THINGS-MEG/preprocessed_newsplit",
#             'fmri': "/home/ldy/fmri_dataset/Preprocessed"
#         }
        
#         # Subject Configurations
#         self.subjects = {
#             'eeg': ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
#             'meg': ['sub-01', 'sub-02', 'sub-03', 'sub-04'],
#             'fmri': ['sub-01', 'sub-02', 'sub-03']
#         }
        
#         self.eval_modality = 'meg'
#         self.modalities = ['eeg', 'meg', 'fmri']
#         self.classes = {
#             'eeg': 200,
#             'meg': 200,
#             'fmri': 100
#         }
        
#         self.output_config = {
#             'output_dir': './outputs/contrast',
#             'project': "train_pos_img_text_rep",
#             'entity': "sustech_rethinkingbci",
#             'name': "lr=3e-4_img_pos_pro_eeg"
#         }
        
#         self.eval_k_values = [2, 4, 10]
    
#     def get_test_config(self):
#         return {
#             'subjects': self.subjects[self.eval_modality],
#             'classes': self.classes[self.eval_modality]
#         }
    
#     def get_device(self):
#         return torch.device(self.device_preference if self.device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
    
#     def setup_environment(self):
#         for key, value in self.env_config.items():
#             os.environ[key] = value
#         wandb.init(mode="disabled")
#         warnings.filterwarnings("ignore")

# def get_dataset(sub, config):
#     """获取数据集，使用正确的路径"""
#     if config.eval_modality == 'eeg':
#         return EEGDataset(config.data_paths['eeg'], adap_subject=sub, subjects=[sub], train=False)
#     elif config.eval_modality == 'meg':
#         from data_preparing.megdatasets_averaged import MEGDataset
#         return MEGDataset(config.data_paths['meg'], subjects=[sub], train=False)
#     elif config.eval_modality == 'fmri':
#         from data_preparing.fmri_datasets_joint_subjects import fMRIDataset
#         return fMRIDataset(config.data_paths['fmri'], adap_subject=sub, subjects=[sub], train=False)
#     else:
#         raise ValueError(f"Unsupported modality: {config.eval_modality}")

# def get_eegfeatures(unified_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
#     unified_model.eval()
#     text_features_all = text_features_all[eval_modality].to(device).float()
#     if eval_modality=='eeg' or eval_modality=='fmri':
#         img_features_all = (img_features_all[eval_modality]).to(device).float()
#     elif eval_modality=='meg':
#         img_features_all = (img_features_all[eval_modality][::12]).to(device).float()  
    
#     total_loss = 0
#     correct = 0
#     top5_correct_count = 0
#     total = 0
#     loss_func = ClipLoss() 
#     all_labels = set(range(text_features_all.size(0)))
#     features_tensor = torch.zeros(0, 0)
    
#     with torch.no_grad():
#         for batch_idx, (modal, data, labels, text, text_features, img, img_features, _, _, sub_ids) in enumerate(dataloader):
#             data = data.to(device)
#             text_features = text_features.to(device).float()
#             labels = labels.to(device)
#             img_features = img_features.to(device).float()
            
#             subject_ids = [extract_id_from_string(sub_id) for sub_id in sub_ids]
#             subject_ids = torch.tensor(subject_ids, dtype=torch.long).to(device)
#             neural_features = unified_model(data, subject_ids, modal=eval_modality)
            
#             logit_scale = unified_model.logit_scale.float()            
#             img_loss = loss_func(neural_features, img_features, logit_scale)
#             total_loss += img_loss.item()
            
#             for idx, label in enumerate(labels):
#                 possible_classes = list(all_labels - {label.item()})
#                 selected_classes = random.sample(possible_classes, k-1) + [label.item()]
#                 selected_img_features = img_features_all[selected_classes]
                
#                 logits_img = logit_scale * neural_features[idx] @ selected_img_features.T
#                 logits_single = logits_img
                
#                 predicted_label = selected_classes[torch.argmax(logits_single).item()]
#                 if predicted_label == label.item():
#                     correct += 1        
#                 if k==test_classes:
#                     _, top5_indices = torch.topk(logits_single, 5, largest=True)
#                     if label.item() in [selected_classes[i] for i in top5_indices.tolist()]:                
#                         top5_correct_count += 1                                 
#                 total += 1              

#     average_loss = total_loss / (batch_idx+1)
#     accuracy = correct / total    
#     top5_acc = top5_correct_count / total    
#     return average_loss, accuracy, top5_acc, labels, features_tensor.cpu()

# def extract_id_from_string(s):
#     match = re.search(r'\d+$', s)
#     if match:
#         return int(match.group())
#     return None

# def print_model_info(model):
#     def format_num(num):
#         for unit in ['','K','M','B','T']:
#             if num < 1000:
#                 return f"{num:.2f}{unit}"
#             num /= 1000
#         return f"{num:.2f}P"
    
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {format_num(total_params)}")
#     print(f"Trainable parameters: {format_num(trainable_params)}")
    
#     if total_params > 0:
#         trainable_percentage = (trainable_params / total_params) * 100
#         print(f"Trainable parameters percentage: {trainable_percentage:.2f}%")
#     else:
#         print("Total parameters count is zero, cannot compute percentage.")

# def run_experiments(config, unified_model, device):
#     test_config = config.get_test_config()
#     results = {
#         'test_accuracies': [],
#         'test_accuracies_top5': [],
#         'v2_accuracies': [],
#         'v4_accuracies': [],
#         'v10_accuracies': []
#     }
    
#     for sub in test_config['subjects']:
#         # Run evaluation for each subject
#         test_dataset = get_dataset(sub, config)
#         test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
        
#         # Collect features
#         text_features_test_all = {config.eval_modality: test_dataset.text_features}
#         img_features_test_all = {config.eval_modality: test_dataset.img_features}
        
#         # Get evaluation results
#         test_loss, test_accuracy, top5_acc, _, _ = get_eegfeatures(
#             unified_model, test_loader, device, text_features_test_all,
#             img_features_test_all, k=test_config['classes'],
#             eval_modality=config.eval_modality, test_classes=test_config['classes']
#         )
        
#         results['test_accuracies'].append(test_accuracy)
#         results['test_accuracies_top5'].append(top5_acc)
        
#         # Evaluate for different k values
#         for k in config.eval_k_values:
#             _, acc, _, _, _ = get_eegfeatures(
#                 unified_model, test_loader, device, text_features_test_all,
#                 img_features_test_all, k=k,
#                 eval_modality=config.eval_modality, test_classes=test_config['classes']
#             )
#             results[f'v{k}_accuracies'].append(acc)
        
#         # Print subject results
#         print(f"\nResults for subject {sub}:")
#         print(f" - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Top5 Accuracy: {top5_acc:.4f}")
#         for k in config.eval_k_values:
#             print(f" - v{k}_acc Accuracy: {results[f'v{k}_accuracies'][-1]:.4f}")
    
#     # Print average results
#     print("\nAverage results across all subjects:")
#     print(f"Test Accuracy: {np.mean(results['test_accuracies']):.4f}")
#     print(f"Test Top5 Accuracy: {np.mean(results['test_accuracies_top5']):.4f}")
#     for k in config.eval_k_values:
#         print(f"v{k}_acc Accuracy: {np.mean(results[f'v{k}_accuracies']):.4f}")

# def main():
#     # Initialize configuration
#     config = ExperimentConfig()
#     config.setup_environment()
    
#     # Get device
#     device = config.get_device()
#     print(f"Using device: {device}")
    
#     try:
#         # Initialize model
#         unified_model = UnifiedEncoder(config.encoder_paths, device)
#         unified_model.load_state_dict(torch.load(config.unified_model_path, map_location=device))
#         unified_model.to(device)
#         unified_model.eval()
#         print("Successfully loaded model")
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         return
    
#     # Print model parameters info
#     print_model_info(unified_model)
    
#     # Run experiments
#     run_experiments(config, unified_model, device)

# if __name__ == "__main__":
#     main()