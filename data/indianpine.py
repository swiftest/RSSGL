from scipy.io import loadmat
from simplecv.data import preprocess
from data.base import FullImageDataset


class NewIndianPinesDataset(FullImageDataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 sample_percent=0.05,
                 batch_size=10):
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        im_mat = loadmat(image_mat_path)
        image = im_mat['indian_pines_corrected']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['indian_pines_gt']

        im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
        im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
        self.vanilla_image = image
        image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
        self.training = training
        self.sample_percent = sample_percent
        self.batch_size = batch_size
        super(NewIndianPinesDataset, self).__init__(image, mask, training, sample_percent=sample_percent, batch_size=batch_size)

    @property
    def num_classes(self):
        return 16
