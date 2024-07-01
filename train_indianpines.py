from simplecv import dp_train as train
import torch
from simplecv.util.logger import eval_progress, speed
import time
from module import RSSGL
from simplecv.util import metric, config, registry
from torch.utils.data.dataloader import DataLoader
from simplecv import registry
from simplecv.core.config import AttrDict
from scipy.io import loadmat
import data.dataloader
from simplecv.module.model_builder import make_model
import torch.nn as nn

import matplotlib.pyplot as plt
from simplecv.data.preprocess import divisible_pad
from simplecv.core._misc import merge_dict
from simplecv.data import preprocess
import numpy as np
from thop import profile


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 5.0 / dpi,
                        ground_truth.shape[0] * 5.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)


def list_to_colormap(x_list, color_map):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = color_map[0]
        if item == 1:
            y[index] = color_map[1]
        if item == 2:
            y[index] = color_map[2]
        if item == 3:
            y[index] = color_map[3]
        if item == 4:
            y[index] = color_map[4]
        if item == 5:
            y[index] = color_map[5]
        if item == 6:
            y[index] = color_map[6]
        if item == 7:
            y[index] = color_map[7]
        if item == 8:
            y[index] = color_map[8]
        if item == 9:
            y[index] = color_map[9]
        if item == 10:
            y[index] = color_map[10]
        if item == 11:
            y[index] = color_map[11]
        if item == 12:
            y[index] = color_map[12]
        if item == 13:
            y[index] = color_map[13]
        if item == 14:
            y[index] = color_map[14]
        if item == 15:
            y[index] = color_map[15]
        if item == 16:
            y[index] = color_map[16]
    return y


def fcn_evaluate_fn(self, test_dataloader, config):

    if self.checkpoint.global_step < 0:
        return
    self._model.eval()
    total_time = 0.
    with torch.no_grad():
        for idx, (im, mask, w) in enumerate(test_dataloader):
            start = time.time()
            y_pred = self._model(im).squeeze()
            torch.cuda.synchronize()
            time_cost = round(time.time() - start, 3)
            y_pred = y_pred.argmax(dim=0).cpu() + 1
            w.squeeze_(dim=0)

            w = w.bool()
            mask = torch.masked_select(mask.view(-1), w.view(-1))
            y_pred = torch.masked_select(y_pred.view(-1), w.view(-1))

            oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
            aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                 self._model.module.config.num_classes,
                                                                 return_accuracys=True)
            kappa = metric.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self._model.module.config.num_classes)
            total_time += time_cost
            speed(self._logger, time_cost, 'im')

            eval_progress(self._logger, idx + 1, len(test_dataloader))

    speed(self._logger, round(total_time / len(test_dataloader), 3), 'batched im (avg)')

    metric_dict = {
        'OA': oa.item(),
        'AA': aa.item(),
        'Kappa': kappa.item()
    }
    for i, acc in enumerate(acc_per_class):
        metric_dict['acc_{}'.format(i + 1)] = acc.item()
    self._logger.eval_log(metric_dict=metric_dict, step=self.checkpoint.global_step)
    
    config['test_oa'].append(oa.item())
    if config['test_oa'][-1] > config['test_oa'][-2]:
        print("test_oa improved from {:.4f} to {:.4f}, saving model ......".format(config['test_oa'][-2], config['test_oa'][-1]))
        torch.save(self._model.state_dict(), config['PATH'])
        config['early_epoch'] = 0
    else:
        print("test_oa did not improve from {:.4f}".format(config['test_oa'][-2]))
        config['early_epoch'] += 1
        config['test_oa'][-1] = config['test_oa'][-2]
        if config['early_epoch'] == config['early_num']:
            self._model.load_state_dict(torch.load(config['PATH']))
            print('Early stopping, the most optimal test_oa is:', config['test_oa'][-1])
            raise StopIteration('Early Stop!')


def register_evaluate_fn(launcher):
    launcher.override_evaluate(fcn_evaluate_fn)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    args = train.parser.parse_args()
    cfg = config.import_config(args.config_path)
    cfg = AttrDict.from_dict(cfg)
    config = merge_dict(cfg['train'], cfg['test'])
    model = make_model(cfg['model'])
    model.to(torch.device('cuda'))
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    try:
        return_dict = train.run(config_path=args.config_path, 
                                model_dir=args.model_dir,
                                cpu_mode=args.cpu,
                                after_construct_launcher_callbacks=[register_evaluate_fn],
                                opts=args.opts)
    except StopIteration:
        model.load_state_dict(torch.load(config['PATH']))
        pass
    else:
        model = return_dict['launcher']._model
        model.load_state_dict(torch.load(config['PATH']))
    model.eval()

    im_mat = loadmat('./IndianPines/Indian_pines_corrected.mat')
    image = im_mat['indian_pines_corrected']
    gt_mat = loadmat('./IndianPines/Indian_pines_gt.mat')
    mask = gt_mat['indian_pines_gt']
    im_cmean = image.reshape((-1, image.shape[-1])).mean(axis=0)
    im_cstd = image.reshape((-1, image.shape[-1])).std(axis=0)
    image = preprocess.mean_std_normalize(image, im_cmean, im_cstd)
    blob = divisible_pad([np.concatenate([image.transpose(2, 0, 1),
                                          mask[None, :, :]], axis=0)], 16, False)
    im = blob[0, :image.shape[-1], :, :]
    im = torch.from_numpy(im).unsqueeze(dim=0).to(torch.device('cuda'))
    
    #print(im.size())
    #flops, params = profile(model, (im,))
    #print('flops: ', flops, 'params: ', params)
    
    gt = mask.flatten()
    y_pred = model(im).squeeze()
    torch.cuda.synchronize()
    y_pred = y_pred.argmax(dim=0).cpu() + 1
            
    outputs = y_pred.numpy()[0:145, 0:145]
    outputs = outputs.flatten()
    for i in range(len(gt)):
        if gt[i] == 0:
            outputs[i] = 0
    color_map = np.array(config['draw']['palette']) / 255.
    y_list = list_to_colormap(outputs, color_map)
    y_gt = list_to_colormap(gt, color_map)
    y_re = np.reshape(y_list, (mask.shape[0], mask.shape[1], 3))
    gt_re = np.reshape(y_gt, (mask.shape[0], mask.shape[1], 3))
    classification_map(y_re, mask, 300, './prediction.pdf')
    classification_map(gt_re, mask, 300, './original.pdf')
