import os
import sys
import re
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

# Set up project paths
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add additional workspace paths
additional_paths = [
    '/mnt/dataset0/ldy/Workspace/EEG_Image_decode_Wrong/Retrieval',
    '/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval',
    '/mnt/dataset0/ldy/Workspace/EEG_Image_decode/Retrieval/model',
    '/mnt/dataset0/ldy/Workspace/FLORA'
]
for path in additional_paths:
    if path not in sys.path:
        sys.path.insert(0, path)

# Configuration dictionary for model settings
MODEL_CONFIG = {
    'model_name': 'FloraW',  # Options: 'ATMS', 'MetaEEG', 'NICE', 'EEGNetv4_Encoder', 'MindEyeModule', etc.
    'mode': 'in_subject',    # Options: 'in_subject' or 'joint'
}

# Import model class based on configuration
model_class_mapping = {
    'ATMS': 'ATMS_retrieval_joint_and_in_train_MEG.ATMS',
    'MetaEEG': 'contrast_retrieval_MEG.MetaEEG',
    'NICE': 'contrast_retrieval_MEG.NICE',
    'EEGNetv4_Encoder': 'contrast_retrieval_MEG.EEGNetv4_Encoder',
    'MindEyeModule': 'contrast_retrieval_MEG.MindEyeModule',
    'MB2CW': 'contrast_retrieval_MEG.MB2CW',
    'CogcapW': 'contrast_retrieval_MEG.CogcapW',
    'MindBridgeW': 'contrast_retrieval_MEG.MindBridgeW',
    'WaveW': 'contrast_retrieval_MEG.WaveW',
    'MedformerNoTSW': 'contrast_retrieval_MEG.MedformerNoTSW',
    'FloraW': 'contrast_retrieval_MEG.FloraW'
}

if MODEL_CONFIG['model_name'] not in model_class_mapping:
    raise ValueError(f"Unknown model type: {MODEL_CONFIG['model_name']}")

# Dynamically import the selected model class
module_name, class_name = model_class_mapping[MODEL_CONFIG['model_name']].rsplit('.', 1)
model_module = __import__(module_name, fromlist=[class_name])
ModelClass = getattr(model_module, class_name)

# Import dataset and loss function
from data_preparing.megdatasets_averaged import MEGDataset
from loss import ClipLoss


def extract_id_from_string(s):
    """Extract numerical subject ID from string.
    
    Args:
        s (str): Input string containing subject ID (e.g., 'sub-01')
        
    Returns:
        int: Extracted numerical ID or None if not found
    """
    match = re.search(r'\d+$', s)
    return int(match.group()) if match else None


def get_megfeatures(sub, meg_model, dataloader, device, text_features_all, img_features_all, k, eval_modality, test_classes):
    """Evaluate MEG features retrieval performance.
    
    Args:
        sub: Subject identifier
        meg_model: Loaded MEG model
        dataloader: DataLoader for evaluation
        device: Device to run computation on
        text_features_all: All text features
        img_features_all: All image features
        k: Number of classes to evaluate against
        eval_modality: Evaluation modality ('meg')
        test_classes: Total number of test classes
        
    Returns:
        tuple: (average_loss, accuracy, top5_acc, labels, features_tensor)
    """
    meg_model.eval()
    text_features_all = text_features_all.to(device).float()
    img_features_all = img_features_all[::12].to(device).float()
    total_loss = 0
    correct = 0
    top5_correct_count = 0
    total = 0
    loss_func = ClipLoss() 
    all_labels = set(range(text_features_all.size(0)))
    save_features = False
    features_list = []
    features_tensor = torch.zeros(0, 0)
    
    with torch.no_grad():
        for batch_idx, (_, data, labels, text, text_features, img, img_features, index, img_index, subject_id) in enumerate(dataloader):
            data = data.to(device)
            text_features = text_features.to(device).float()
            labels = labels.to(device)
            img_features = img_features.to(device).float()
            
            batch_size = data.size(0) 
            subject_id = extract_id_from_string(subject_id[0])
            subject_ids = torch.full((batch_size,), subject_id, dtype=torch.long).to(device)
            neural_features = meg_model(data)
            
            logit_scale = meg_model.logit_scale.float()            
            features_list.append(neural_features)
               
            img_loss = loss_func(neural_features, img_features, logit_scale)
            loss = img_loss        
            total_loss += loss.item()
            
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

        if save_features:
            features_tensor = torch.cat(features_list, dim=0)
            print("features_tensor", features_tensor.shape)
            torch.save(features_tensor.cpu(), f"ATM_S_neural_features_{sub}_train.pt")
            
    average_loss = total_loss / (batch_idx+1)
    accuracy = correct / total    
    top5_acc = top5_correct_count / total    
    return average_loss, accuracy, top5_acc, labels, features_tensor.cpu()


# ======================================== Configuration =============================================
test_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
device_preference = 'cuda:4'
device_type = 'gpu'
data_path = "/home/ldy/THINGS-MEG/preprocessed_newsplit"
device = torch.device(device_preference if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# ======================================== Configuration =============================================

# Experiment setup
mode = MODEL_CONFIG['mode']
test_classes = 200
eval_modality = 'meg'

# Initialize lists for storing accuracies
test_accuracies = []
test_accuracies_top5 = []
v2_accuracies = []
v4_accuracies = []
v10_accuracies = []

# Initialize dictionary to store results per subject
subject_results = {sub: {} for sub in test_subjects}

print("\n" + "="*80)
print(f"Starting experiment with following configuration:")
print(f"Model: {MODEL_CONFIG['model_name']}")
print(f"Mode: {mode}")
print(f"Test subjects: {', '.join(test_subjects)}")
print(f"Number of test classes: {test_classes}")
print(f"Evaluation modality: {eval_modality}")
print("="*80 + "\n")

# Main evaluation loop
for sub in test_subjects:
    print(f"\nProcessing subject: {sub}")
    
    # Load appropriate model based on mode
    meg_model = ModelClass()
    base_path = f"/mnt/dataset1/ldy/Workspace/FLORA/models/{MODEL_CONFIG['model_name']}"
    
    if mode == 'joint':
        across_dir = os.path.join(base_path, 'across', 'MEG')
        time_folder = os.listdir(across_dir)[0]
        model_path = os.path.join(across_dir, time_folder, '40.pth')
        print(f"Loading joint model from: {model_path}")
    else:  # in_subject mode
        subject_num = sub.split('-')[1]  # Extract subject number (e.g., "01" from "sub-01")
        subject_dir = os.path.join(base_path, 'in_subject', 'MEG', f'sub-{subject_num}')
        time_folder = os.listdir(subject_dir)[0]
        model_path = os.path.join(subject_dir, time_folder, '40.pth')
        print(f"Loading in-subject model from: {model_path}")
    
    meg_model.load_state_dict(torch.load(model_path, map_location=device))
    meg_model.to(device)
    meg_model.eval()

    # Setup dataset and dataloader
    test_dataset = MEGDataset(data_path, adap_subject=sub, subjects=test_subjects, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    
    text_features_test_all = test_dataset.text_features    
    img_features_test_all = test_dataset.img_features
    
    # Run evaluations with different k values
    test_loss, test_accuracy, top5_acc, labels, meg_features_test = get_megfeatures(
        sub, meg_model, test_loader, device, text_features_test_all, img_features_test_all, 
        k=test_classes, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v2_acc, _, _, _ = get_megfeatures(
        sub, meg_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=2, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v4_acc, _, _, _ = get_megfeatures(
        sub, meg_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=4, eval_modality=eval_modality, test_classes=test_classes
    )
    _, v10_acc, _, _, _ = get_megfeatures(
        sub, meg_model, test_loader, device, text_features_test_all, img_features_test_all,
        k=10, eval_modality=eval_modality, test_classes=test_classes
    )    
    
    # Store results
    test_accuracies.append(test_accuracy)
    test_accuracies_top5.append(top5_acc)
    v2_accuracies.append(v2_acc)
    v4_accuracies.append(v4_acc)
    v10_accuracies.append(v10_acc)
    
    # Store individual results
    subject_results[sub] = {
        'test_acc': test_accuracy,
        'top5_acc': top5_acc,
        'v2_acc': v2_acc,
        'v4_acc': v4_acc,
        'v10_acc': v10_acc
    }
    
    print(f"\nResults for {sub}:")
    print(f" - Test Accuracy: {test_accuracy:.4f}")
    print(f" - Top5 Accuracy: {top5_acc:.4f}")    
    print(f" - v2 Accuracy: {v2_acc:.4f}")
    print(f" - v4 Accuracy: {v4_acc:.4f}")
    print(f" - v10 Accuracy: {v10_acc:.4f}")

# Print final summary
print("\n" + "="*80)
print(f"EXPERIMENT SUMMARY")
print(f"Evaluation modality: {eval_modality}")
print(f"Model: {MODEL_CONFIG['model_name']}")
print(f"Mode: {mode}")
print(f"Subjects: {', '.join(test_subjects)}")
print("="*80)

print("\nOverall Performance:")
print(f"Test Accuracy: {np.mean(test_accuracies):.4f} ± {np.std(test_accuracies):.4f}")
print(f"Top5 Accuracy: {np.mean(test_accuracies_top5):.4f} ± {np.std(test_accuracies_top5):.4f}")
print(f"v2 Accuracy: {np.mean(v2_accuracies):.4f} ± {np.std(v2_accuracies):.4f}")
print(f"v4 Accuracy: {np.mean(v4_accuracies):.4f} ± {np.std(v4_accuracies):.4f}")
print(f"v10 Accuracy: {np.mean(v10_accuracies):.4f} ± {np.std(v10_accuracies):.4f}")