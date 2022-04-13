from torch.utils.data.dataloader import DataLoader
from simplecv import registry
from data.base import MinibatchSampler
from data.pavia import NewPaviaDataset
from data.indianpine import NewIndianPinesDataset
from data.salinas import NewSalinasDataset


@registry.DATALOADER.register('NewPaviaLoader')
class NewPaviaLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewPaviaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.sample_percent, self.batch_size)
        
        sampler = MinibatchSampler(dataset)
        super(NewPaviaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )
        
        
@registry.DATALOADER.register('NewIndianPinesLoader')
class NewIndianPinesLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewIndianPinesDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                        self.sample_percent, self.batch_size)
        sampler = MinibatchSampler(dataset)
        super(NewIndianPinesLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.05,
            batch_size=10)
        )
        
        
@registry.DATALOADER.register('NewSalinasLoader')
class NewSalinasLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewSalinasDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.sample_percent, self.batch_size)
        sampler = MinibatchSampler(dataset)
        super(NewSalinasLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )