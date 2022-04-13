from torch.utils.data import dataset
import numpy as np
from simplecv.data.preprocess import divisible_pad
import torch
from torch.utils import data


class FullImageDataset(dataset.Dataset):
    def __init__(self,
                 image,
                 mask,
                 training,
                 sample_percent=0.01,
                 batch_size=10,
                 ):
        self.image = image  # Normalized data used as input（610, 340, 103）
        self.mask = mask  # (610, 340)
        self.training = training
        self.sample_percent = sample_percent
        self.batch_size = batch_size
        self.preset()

    def preset(self):
        train_indicator, test_indicator= fixed_num_sample(self.mask, self.sample_percent, self.num_classes)

        blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :]], axis=0)], 16, False)
        
        im = blob[0, :self.image.shape[-1], :, :]  # (103, 624, 352)
        mask = blob[0, -3, :, :]  # (624, 352)
        self.train_indicator = blob[0, -2, :, :]  # (624, 352)
        self.test_indicator = blob[0, -1, :, :]  # (624, 352)
        if self.training:
            self.train_inds_list = minibatch_sample(mask, self.train_indicator, self.batch_size)
        self.pad_im = im
        self.pad_mask = mask

    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.train_indicator, self.batch_size)

    @property
    def num_classes(self):
        return 9

    def __getitem__(self, idx):
        if self.training:
            return self.pad_im, self.pad_mask, self.train_inds_list[idx]

        else:
            return self.pad_im, self.pad_mask, self.test_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)
        else:
            return 1


class MinibatchSampler(data.Sampler):
    def __init__(self, dataset: FullImageDataset):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()

    def __iter__(self):
        self.dataset.resample_minibatch()
        n = len(self.dataset)
        return iter(torch.randperm(n, generator=self.g).tolist())

    def __len__(self):
        return len(self.dataset)


def fixed_num_sample(gt_mask: np.ndarray, sample_percent, num_classes):
    """
    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
    Returns:
        train_indicator, test_indicator
    """
    #(106，610，340)
    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    for i in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == i)[0]
        count=np.sum(gt_mask_flatten == i)
        #print(count)
        num_train_samples=np.ceil(count*sample_percent)
        num_train_samples = num_train_samples.astype(np.int32)
        if num_train_samples <5:
            num_train_samples=5  # At least 5 samples per class
        np.random.shuffle(inds)

        train_inds = inds[:num_train_samples]
        test_inds = inds[num_train_samples:]
        #print(len(test_inds))
        #print(len(train_inds))  # Print how many training samples there are for each class

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1

    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)
    return train_indicator, test_indicator


def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, batch_size):
    """
    Args:
        gt_mask: 2-D array of shape [height, width]
        train_indicator: 2-D array of shape [height, width]
        minibatch_size:
    Returns:
    """
    # split into N classes
    cls_list = np.unique(gt_mask)
    inds_dict_per_class = dict()
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        np.random.shuffle(inds)

        inds_dict_per_class[cls] = inds

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
            np.random.shuffle(inds)
            cd=min(batch_size, len(inds))
            fetch_inds = inds[:cd]
            train_inds[fetch_inds] = 1

        cnt += 1
        if cnt == 11:
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))  # Take 10 training samples for each class each time
