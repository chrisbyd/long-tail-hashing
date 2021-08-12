import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from collections import defaultdict
from data.transform import train_transform, query_transform, Onehot, encode_onehot, TwoCropTransform, train_two_transform
import random

def load_data(root, batch_size, num_workers):
    """
    Load cifar-10 dataset.

    Args
        root(str): Path of dataset.
        batch_size(int): Batch size.
        num_workers(int): Number of data loading workers.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    root = os.path.join(root, 'images')
    num_classes = 100
    train_dataloader = DataLoader(
        ImagenetDataset(
            os.path.join(root, 'train'),
            transform=train_transform(),
            contrast_transform= TwoCropTransform(train_two_transform),
            target_transform=Onehot(num_classes),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
    )

    query_dataloader = DataLoader(
        ImagenetDataset(
            os.path.join(root, 'query'),
            transform=query_transform(),
            contrast_transform=query_transform(),
            target_transform=Onehot(num_classes),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    retrieval_dataloader = DataLoader(
        ImagenetDataset(
            os.path.join(root, 'database'),
            transform=query_transform(),
            contrast_transform=query_transform(),
            target_transform=Onehot(num_classes),
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, query_dataloader, retrieval_dataloader,


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, contrast_transform= None, target_transform=None):
        self.root = root
        self.transform = transform
        self.contrast_transform = contrast_transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            cur_class = os.path.join(self.root, cl)
            files = os.listdir(cur_class)
            files = [os.path.join(cur_class, i) for i in files]
            self.data.extend(files)
            self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        self.targets = np.asarray(self.targets)
        self.class_to_indexes = self._get_class_dict()
        self.class_weight, self.sum_weight = self._get_weight()
        self.onehot_targets = torch.from_numpy(encode_onehot(self.targets, 100)).float()

    def get_onehot_targets(self):
        return self.onehot_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.contrast_transform is not None:
            img = self.contrast_transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        meta = dict()
        sample_class = self.sample_class_index_by_weight()
        sample_indexes = self.class_to_indexes[sample_class]
        sample_index = random.choice(sample_indexes)

        sample_image, sample_target = self.data[sample_index], self.targets[sample_index]
        sample_image = Image.open(sample_image).convert('RGB')
        if self.transform is not None:
            sample_image = self.transform(sample_image)
        if self.target_transform is not None:
            sample_target = self.target_transform(sample_target)
        
        meta['sample_image'] = sample_image
        meta['sample_label'] = sample_target


        return img, target, meta,item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def _get_class_dict(self):
        class_dict = defaultdict(list)

        for index, target in enumerate(self.targets):
            class_dict[target].append(index)
        return class_dict
    
    def _get_weight(self):
        num_classes = len(ImagenetDataset.classes)
        num_list = [0] * num_classes
        for target in self.targets:
            num_list[target] += 1
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight
    
    def sample_class_index_by_weight(self):
        num_classes = len(ImagenetDataset.classes)
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i
