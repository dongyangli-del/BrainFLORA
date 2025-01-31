try:
    from data_preparing.eegdatasets import EEGDataset
    from data_preparing.megdatasets_averaged import MEGDataset   
    from data_preparing.fmri_datasets_joint_subjects import fMRIDataset             
except ModuleNotFoundError:
    from eegdatasets import EEGDataset  
    from megdatasets_averaged import MEGDataset 
    from fmri_datasets_joint_subjects import fMRIDataset
import torch
import pickle
from torch.utils.data import DataLoader
import itertools
import torch.nn.functional as F
import pickle
from torch.utils.data import DataLoader, Sampler
import itertools
import torch.nn.functional as F
import random
import math

import torch
import random
import math
from torch.utils.data import Sampler

class PartialStratifiedBatchSampler(Sampler):
    """
    Partial Stratified Batch Sampler that ensures each batch contains 
    repeated samples of the same class.

    Args:
        labels (list or tensor): A list of labels corresponding to the dataset.
        batch_size (int): Number of samples in each batch.
        samples_per_class (int): Number of samples for each class in a batch.

    Returns:
        Batches of indices where labels are stratified and repeated.
    """
    def __init__(self, labels, batch_size, samples_per_class):
        # Validate input
        if samples_per_class <= 0:
            raise ValueError("samples_per_class must be positive.")
        if batch_size % samples_per_class != 0:
            raise ValueError("batch_size must be divisible by samples_per_class.")

        self.labels = labels
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

        # Map labels to their corresponding indices
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        # All unique labels in the dataset
        self.labels_set = list(self.label_to_indices.keys())

        # Number of classes per batch
        self.classes_per_batch = min(len(self.labels_set), self.batch_size // self.samples_per_class)

        # Ensure classes_per_batch is valid
        if self.classes_per_batch <= 0:
            raise ValueError("classes_per_batch must be positive. Adjust batch_size or samples_per_class.")
        if len(self.labels_set) < self.classes_per_batch:
            raise ValueError("The number of unique classes is smaller than the required classes_per_batch. Reduce batch_size or samples_per_class.")

        # Shuffle the indices within each class initially
        for cls in self.labels_set:
            random.shuffle(self.label_to_indices[cls])

    def __iter__(self):
        while True:
            batch = []
            # Randomly select classes for this batch
            selected_classes = random.sample(self.labels_set, self.classes_per_batch)

            for cls in selected_classes:
                for _ in range(self.samples_per_class):
                    # Randomly select an index from the current class
                    idx = random.choice(self.label_to_indices[cls])
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []

            # Handle any remaining samples (if necessary)
            if len(batch) > 0:
                yield batch

    def __len__(self):
        # Total number of batches
        return math.ceil(len(self.labels) / self.batch_size)


class MetaEEGDataset():
    def __init__(self, eeg_data_path, eeg_subjects, train=True, use_caption=False):
        self.eeg_data_path = eeg_data_path
        self.eeg_subjects = eeg_subjects
        self.n_cls = 1654 if train else 200
        self.train = train
        eeg_data = None
        self.modal = 'eeg'
        sub_eeg_dataset = EEGDataset(eeg_data_path, subjects=eeg_subjects, train=train, use_caption=use_caption)
        self.text_features = sub_eeg_dataset.text_features
        self.img_features = sub_eeg_dataset.img_features
        self.labels = sub_eeg_dataset.labels
        self.text = sub_eeg_dataset.text
        self.img = sub_eeg_dataset.img
        self.eeg_data = sub_eeg_dataset.data
        

    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        eeg_data = self.eeg_data[index]

        index_n_sub_train = self.n_cls * 10 * 4
        index_n_sub_test = self.n_cls * 1 * 80

        if self.train:
            label = self.labels[index]
        else:
            label = self.labels[index]
            
        # text_index: classes
        if self.train:
            text_index = (index % index_n_sub_train) // (10 * 4)
        else:
            text_index = (index % index_n_sub_test)
        # img_index: classes * 10
        if self.train:
            img_index = (index % index_n_sub_train) // (4)
        else:
            img_index = (index % index_n_sub_test)    
        # print("text_index", text_index)
        # print("self.text", self.text)
        text = self.text[text_index]
        
        img = self.img[img_index]

        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]

        return self.modal, eeg_data, label, text, text_features, img, img_features, index, img_index, 'sub--1'
    
    def __len__(self):
        return self.eeg_data.shape[0]


class MetaMEGDataset():
    def __init__(self, meg_data_path, meg_subjects, train=True, use_caption=False):
        self.meg_data_path = meg_data_path
        self.meg_subjects = meg_subjects
        self.n_cls = 1654 if train else 200
        self.train = train
        meg_data = None
        self.modal = 'meg'
        sub_meg_dataset = MEGDataset(meg_data_path, subjects=meg_subjects, train=train, use_caption=use_caption)
        self.text_features = sub_meg_dataset.text_features
        self.img_features = sub_meg_dataset.img_features
        self.labels = sub_meg_dataset.labels
        self.text = sub_meg_dataset.text
        self.img = sub_meg_dataset.img
        self.meg_data = sub_meg_dataset.data
               
    
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 12 * 1)
        meg_data = self.meg_data[index]
        
        index_n_sub_train = self.n_cls * 12 * 1
        index_n_sub_test = self.n_cls * 1 * 12        
        
        if self.train:
            label = self.labels[index]
        else:
            label = self.labels[index]
        
        # text_index: classes
        if self.train:
            text_index = (index % index_n_sub_train) // (12 * 1)
        else:
            text_index = (index % index_n_sub_test)
        # img_index: classes * 10
        if self.train:
            img_index = (index % index_n_sub_train)
        else:
            img_index = (index % index_n_sub_test)
        # try:
        text = self.text[text_index]
        img = self.img[img_index]
        # except:
        #     print("text_index", text_index)
        #     print("self.text", len(self.text))

        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        return self.modal, meg_data, label, text, text_features, img, img_features, index, img_index, 'sub--1'
    
    def __len__(self):
        return self.meg_data.shape[0]

class MetafMRIDataset():
    def __init__(self, fmri_data_path, fmri_subjects, train=True, use_caption=False):
        self.fmri_data_path = fmri_data_path
        self.fmri_subjects = fmri_subjects
        self.n_cls = 720 if train else 100
        self.train = train        
        self.modal = 'fmri'
        sub_fmri_dataset = fMRIDataset(fmri_data_path, subjects=fmri_subjects, train=train, use_caption=use_caption)
        self.text_features = sub_fmri_dataset.text_features
        self.img_features = sub_fmri_dataset.img_features
        self.labels = sub_fmri_dataset.labels
        self.text = sub_fmri_dataset.text
        self.img = sub_fmri_dataset.img
        self.fmri_data = sub_fmri_dataset.data               

        # Calculate the length of data for each subject     
           
        self.subject_data_lens = [data.shape[0] for data in self.fmri_data]
        # print("self.subject_data_lens", self.subject_data_lens)
        self.cumulative_data_lens = [0] + list(itertools.accumulate(self.subject_data_lens))  # Cumulative lengths for indexing

    def __getitem__(self, index):
        # Step 1: Find which subject the index belongs to
        subject_idx = None
        for i, cum_len in enumerate(self.cumulative_data_lens[1:]):
            if index < cum_len:
                subject_idx = i
                break
        subject_offset = index - self.cumulative_data_lens[subject_idx]
        
        # Step 2: Get the data and label for the specific subject and offset
        x = self.fmri_data[subject_idx][subject_offset]
        label = self.labels[subject_idx][subject_offset]
        
        # Get the subject ID from fmri_subjects list
        subject_id = self.fmri_subjects[subject_idx]  # 获取被试标识符
        # print(f"Index: {index}, Subject Index: {subject_idx}, Subject Offset: {subject_offset}")
        # print(f"Cumulative lengths: {self.cumulative_data_lens}")

        # Pad the fmri data (x) to 7000 if necessary
        target_length = 7000
        if x.shape[0] < target_length:
            # If x is smaller than the target size, pad it
            padding_size = target_length - x.shape[0]
            x = F.pad(x, (0, padding_size), value=0)  # Pad at the end with zeros
        elif x.shape[0] > target_length:
            # If x is larger than the target size, truncate it
            x = x[:target_length]

        # Step 3: Calculate text and img indices based on subject-specific parameters
        index_n_sub_train = self.n_cls * 12 * 1
        index_n_sub_test = self.n_cls * 12 * 1

        if self.train:
            text_index = (subject_offset % index_n_sub_train) // (12 * 1)
            img_index = (subject_offset % index_n_sub_train)
        else:
            text_index = (subject_offset % index_n_sub_test)
            img_index = (subject_offset % index_n_sub_test)

        # 获取文本、图像和特征
        text = self.text[text_index]
        img = self.img[img_index]
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]

        return self.modal, x, label, text, text_features, img, img_features, index, img_index, subject_id
        
    def __len__(self):
        return sum(self.subject_data_lens)  # 所有被试的数据总长度


class MetaDataLoader:
    def __init__(self, eeg_dataset=None, meg_dataset=None, fmri_dataset=None, batch_size=32, is_shuffle_batch=True, shuffle=True, drop_last=False, modalities=['eeg', 'meg', 'fmri']):
        self.is_shuffle_batch = is_shuffle_batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.modalities = modalities

        # Map modalities to their corresponding datasets
        self.datasets = {
            'eeg': eeg_dataset,
            'meg': meg_dataset,
            'fmri': fmri_dataset
        }

        # Initialize DataLoaders for the selected modalities
        self.loaders = {}
        for modality in self.modalities:
            dataset = self.datasets.get(modality)
            if dataset is not None:
                loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size if self.is_shuffle_batch else self.batch_size // len(self.modalities),
                    shuffle=self.shuffle,
                    drop_last=drop_last
                )
                self.loaders[modality] = loader
            else:
                raise ValueError(f"No dataset provided for modality '{modality}'")

    def __iter__(self):
        # Reset the iterators at the start of each epoch
        self.iters = {modality: iter(loader) for modality, loader in self.loaders.items()}
        if self.is_shuffle_batch:
            self.modality_list = list(self.modalities)
            self.current_modality_index = 0
        return self
    def __len__(self):
        total_samples = 0
        for modality in self.modalities:
            dataset = self.datasets.get(modality)
            if dataset is not None:
                total_samples += len(dataset)
        return total_samples
    def __next__(self):
        if self.is_shuffle_batch:
            if not self.modality_list:
                raise StopIteration
            modality = self.modality_list[self.current_modality_index]
            try:
                batch_data = next(self.iters[modality])
                self.current_modality_index = (self.current_modality_index + 1) % len(self.modality_list)
                return batch_data
            except StopIteration:
                # Remove exhausted modality
                del self.iters[modality]
                self.modality_list.remove(modality)
                if not self.modality_list:
                    raise StopIteration
                self.current_modality_index = self.current_modality_index % len(self.modality_list)
                return self.__next__()
        else:
            try:
                batch_elements = []
                for modality in self.modalities:
                    batch = next(self.iters[modality])
                    batch_elements.extend(batch)
                return tuple(batch_elements)
            except StopIteration:
                raise StopIteration



if __name__ == '__main__':
    eeg_data_path = "/home/ldy/4090_Workspace/4090_THINGS/Preprocessed_data_250Hz"
    meg_data_path = "/home/ldy/THINGS-MEG/preprocessed_newsplit"
    fmri_data_path = "/home/ldy/fmri_dataset/Preprocessed"
    # eeg_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    eeg_subjects = ['sub-01']
    # meg_subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']
    meg_subjects = ['sub-01']
    fmri_subjects = ['sub-01']
    # 初始化数据集
    eegdataset = MetaEEGDataset(eeg_data_path, eeg_subjects, train=True)
    megdataset = MetaMEGDataset(meg_data_path, meg_subjects, train=True)
    fmridataset = MetafMRIDataset(fmri_data_path, meg_subjects, train=True)

    # 创建 MetaDataLoader
    metadataloader = MetaDataLoader(
        eeg_dataset=eegdataset, 
        meg_dataset=megdataset, 
        fmri_dataset = fmridataset,
        batch_size=64,         
    )

    # 迭代数据加载器
    for batch_idx, batch_data in enumerate(metadataloader):
        if isinstance(batch_data, tuple):
            # 处理多模态数据
            for modality_data in batch_data:
                modal, data, labels, text, text_features, img, img_features, index, img_index, subject_id = modality_data
                print(f"Batch {batch_idx + 1} - Modality: {modal}")
                print(f" - Data shape: {data.shape}")
                print(f" - Labels: {labels}")
                print(f" - Text features shape: {text_features.shape}")
                print(f" - Image features shape: {img_features.shape}")
                print(f" - Subject ID: {subject_id}")
        else:
            # 处理单一模态数据
            modal, data, labels, text, text_features, img, img_features, index, img_index, subject_id = batch_data
            print(f"Batch {batch_idx + 1} - Modality: {modal}")
            print(f" - Data shape: {data.shape}")
            print(f" - Labels: {labels}")
            print(f" - Text features shape: {text_features.shape}")
            print(f" - Image features shape: {img_features.shape}")
            print(f" - Subject ID: {subject_id}")
        # 示例：只迭代前 2 个批次
        if batch_idx >= 5:
            break