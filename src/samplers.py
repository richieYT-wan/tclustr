import torch
from torch.utils.data import Sampler, SequentialSampler, RandomSampler
import numpy as np
from itertools import cycle, islice


class MinorityClassSampler(Sampler):
    """
    Custom Superclass to handle saving labels, minority classes, pepmap, indices as attributes as well as a counter
    """

    def __init__(self, pepmap, labels, batch_size, minority_classes):
        super(MinorityClassSampler, self).__init__()
        self.pepmap = pepmap
        self.inverse_pepmap = {v: k for k, v in pepmap.items()}
        self.labels = labels
        self.minority_classes = minority_classes
        self.class_cycle = cycle(minority_classes)
        self.batch_size = batch_size
        self.num_batches = len(labels) // batch_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        raise NotImplementedError


class ClassSpecificBatchSampler(MinorityClassSampler):
    def __init__(self, pepmap, labels, batch_size, minority_classes):
        super(ClassSpecificBatchSampler, self).__init__(pepmap, labels, batch_size, minority_classes)

        # Create a mapping from each minority class to its indices
        self.class_indices = {cls: np.where(self.labels == cls)[0] for cls in minority_classes}
        # Create a cycle of the class labels to rotate which class is used in each batch
        self.class_cycle = cycle(minority_classes)

    def __iter__(self):
        for _ in range(self.num_batches):
            current_class = next(self.class_cycle)  # Continues from where it left off
            indices = (self.labels == current_class).nonzero(as_tuple=True)[0]
            indices = indices[torch.randperm(len(indices))]  # Shuffle indices
            yield indices[:self.batch_size].tolist()




