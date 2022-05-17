import torch.nn as nn
import torch.nn.functional as F
from simplecv.interface import CVModule
from simplecv import registry
import torch
from torch.autograd import Variable
import numpy as np
import bisect

import scipy.io as io
from simplecv import dp_train as train


args = train.parser.parse_args()
config_path = args.config_path


global kernel_shape
global loss_weight


if config_path == "RSSGL.RSSGL_Pavia":
    kernel_shape = (3, 5, 5)
    loss_weight = 0.005
elif config_path == "RSSGL.RSSGL_Salinas":
    kernel_shape = (3, 5, 5)
    loss_weight = 0.001
elif config_path == "RSSGL.RSSGL_Indian_Pines":
    kernel_shape = (3, 3, 3)
    loss_weight = 0.0009


def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(inplace=True),
    )


def gn_relu(in_channel, num_group):
    return nn.Sequential(
        nn.GroupNorm(num_group, in_channel),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 2, 1),
        nn.ReLU(inplace=True)
    )


def repeat_block(block_channel1, r, n):
    cl_channel = block_channel1 / 8
    cl_channel = int(cl_channel)
    cl2_channel = int(cl_channel / 2)
    gn_a = int(block_channel1 / 2)
    layers = (
        nn.Sequential(
            Si_ConvLSTM(input_channels=1, hidden_channels=[4, 4], kernel_size=kernel_shape, step=8,
                        effective_step=7).cuda(),
            BasicBlock(gn_a), 
            gn_relu(block_channel1, r)
        )
    )
    return nn.Sequential(*layers)


@registry.MODEL.register('RSSGL')
class RSSGL(CVModule):
    def __init__(self, config):
        super(RSSGL, self).__init__(config)
        r = int(4 * self.config.reduction_ratio)
        block1_channels = int(self.config.block_channels[0] * self.config.reduction_ratio / r) * r
        block2_channels = int(self.config.block_channels[1] * self.config.reduction_ratio / r) * r
        block3_channels = int(self.config.block_channels[2] * self.config.reduction_ratio / r) * r
        block4_channels = int(self.config.block_channels[3] * self.config.reduction_ratio / r) * r

        self.feature_ops = nn.ModuleList([
            conv3x3_gn_relu(self.config.in_channels, block1_channels, r),

            repeat_block(block1_channels, r, self.config.num_blocks[0]),  # num_blocks=(1, 1, 1, 1)
            nn.Identity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(block2_channels, r, self.config.num_blocks[1]),
            nn.Identity(),
            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels, r, self.config.num_blocks[2]),
            nn.Identity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(block4_channels, r, self.config.num_blocks[3]),
            nn.Identity(),
        ])
        inner_dim = int(self.config.inner_dim * self.config.reduction_ratio)

        self.BasicBlock_list = nn.ModuleList([
            BasicBlock(inner_dim),
            BasicBlock(inner_dim),
            BasicBlock(inner_dim),
            BasicBlock(inner_dim),
        ])
        self.spation_list = nn.ModuleList([
            SpatialAttention(),
            SpatialAttention(),
            SpatialAttention(),
            SpatialAttention(),
        ])
        self.reduce_1x1convs = nn.ModuleList([
            nn.Conv2d(block1_channels, inner_dim, 1),
            nn.Conv2d(block2_channels, inner_dim, 1),
            nn.Conv2d(block3_channels, inner_dim, 1),
            nn.Conv2d(block4_channels, inner_dim, 1),
        ])
        self.fuse_3x3convs = nn.ModuleList([
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            conv3x3_gn_relu(inner_dim, inner_dim, r),
            nn.Conv2d(inner_dim, self.config.in_channels, 3, 1, 1),
        ])

        self.cls_pred_conv = nn.Conv2d(self.config.in_channels, self.config.num_classes, 1)

    def top_down(self, top, lateral):
        top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
        return lateral + top2x

    def forward(self, x, y=None, train_inds=None, **kwargs):
        feat_list = []
        for op in self.feature_ops:
            x = op(x)

            if isinstance(op, nn.Identity):
                feat_list.append(x)
        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]

        inner_feat_list.reverse()  # [(batch_size, 128, 78, 44), (batch_size, 128, 156, 88), ...]
        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]  # (batch_size, 128, 78, 44)
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])

            out = self.fuse_3x3convs[i + 1](inner)

            out_feat_list.append(out)
        final_feat = out_feat_list[-1]  # (batch_size, 103, 624, 352) This is the final feature space!!!
        #mat_path = './feature.mat'
        #mat = final_feat.cpu().detach().numpy()
        #io.savemat(mat_path, {'name': mat})
        logit = self.cls_pred_conv(final_feat)  # (batch_size, 9, 624, 352)
        if self.training:
            loss_dict = {'cls_loss': self.loss(logit, y, train_inds, final_feat)}
            return loss_dict

        return torch.softmax(logit, dim=1)  # (batch_size, 9, 624, 352)

    def loss(self, x, y, train_inds, final_feat):
        beta = 0.9999
        if config_path == "RSSGL.RSSGL_Pavia":
            cls_num_list = [67, 187, 21, 31, 14, 51, 14, 37, 10]
        elif config_path == "RSSGL.RSSGL_Salinas":
            cls_num_list = [21, 38, 20, 14, 27, 40, 36, 113, 63, 33, 11, 20, 10, 11, 73, 19]
        elif config_path == "RSSGL.RSSGL_Indian_Pines":
            cls_num_list = [5, 72, 42, 12, 25, 37, 5, 24, 5, 49, 123, 30, 11, 64, 20, 5]
        else:
            print("no cls_num_list")
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        # In a labeled sample, the smaller the number of categories, the greater its loss weight.
        per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
        
        # (1, 624, 352) Attention!!! The program here is very critical. Why calculate the loss like this?
        losses = F.cross_entropy(x, y.long() - 1, ignore_index=-1, reduction='none', weight=per_cls_weights)
        weighted_losses = losses.mul(train_inds).sum() / train_inds.sum()  
        losses = weighted_losses + loss_weight * self.statistical_loss(y, train_inds, final_feat)
        return losses
    
    def improved_triplet_loss(self, y, train_inds, final_feat):
        alpha = 0.2
        y = y.squeeze()  # (624, 352)
        train_inds = train_inds.squeeze()  #(624, 352)
        final_feat = final_feat.squeeze()  # (103, 624, 352)
        
        cls_list = torch.unique(y)  # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        num_cls = len(cls_list) - 1
        num_train = int(train_inds.sum())
        feat_dimension = final_feat.size()[0]  # 103
        
        location = torch.where(train_inds == 1.)
        label = y[location]
        
        feat_dict_per_class = dict()
        train_per_class = []
        for i in range(1, num_cls+1):
            feat_inds = torch.where(label==i)[0]
            train_per_class.append(len(feat_inds))
            feat_dict_per_class[i] = final_feat[:, location[0][feat_inds], location[1][feat_inds]]
        
        feat = feat_dict_per_class[1]
        for i in range(1, num_cls):
            feat = torch.cat([feat, feat_dict_per_class[i+1]], dim=1)
        
        distance = torch.zeros((num_train, num_train), device=torch.device("cuda:0"))
        for i in range(num_train):
            anchor = torch.unsqueeze(feat[:, i], 1)
            distance[i, :] = torch.norm(anchor - feat, p=2, dim=0)
        
        acc_index = np.zeros(num_cls).astype(np.int16)
        for i in range(num_cls):
            acc_index[i] = sum(train_per_class[:i+1])
        
        result = torch.zeros(num_train).cuda()
        for i in range(num_train):
            position = bisect.bisect(acc_index, i)
            if position == 0:
                left_index = 0
            else:
                left_index = acc_index[position-1]
            right_index = acc_index[position]
            #print(left_index, right_index)
            intra_dist = distance[i, left_index:right_index]
            inter_dist = torch.cat([distance[i, :left_index], distance[i, right_index:]])
            #print(intra_dist.size(), inter_dist.size())
            #print(torch.topk(intra_dist, k=4)[0], torch.topk(inter_dist, k=4, largest=False)[0])
            positive = (torch.topk(intra_dist, k=4)[0]).sum() / 4
            negative = (torch.topk(inter_dist, k=4, largest=False)[0]).sum() / 4
            result[i] = alpha + positive - negative
            if result[i] < 0.0:
                result[i] = 0.0
        #print(result) 
        return result.sum()
    
    def sphere_loss(self, y, train_inds, final_feat):
        tan_beta_2 = 0.2
        y = y.squeeze()  # (624, 352)
        train_inds = train_inds.squeeze()  #(624, 352)
        final_feat = final_feat.squeeze()  # (103, 624, 352)
        
        cls_list = torch.unique(y)  # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        num_cls = len(cls_list) - 1
        num_train = int(train_inds.sum())
        feat_dimension = final_feat.size()[0]  # 103
        
        location = torch.where(train_inds == 1.)
        label = y[location]
        
        feat_dict_per_class = dict()
        train_per_class = []
        for i in range(1, num_cls+1):
            feat_inds = torch.where(label==i)[0]
            train_per_class.append(len(feat_inds))
            feat_dict_per_class[i] = final_feat[:, location[0][feat_inds], location[1][feat_inds]]
        
        feat = feat_dict_per_class[1]
        for i in range(1, num_cls):
            feat = torch.cat([feat, feat_dict_per_class[i+1]], dim=1)
            
        ck = dict()
        for i in range(1, num_cls+1):
            ck[i] = feat_dict_per_class[i].mean(dim=1).unsqueeze(dim=1)
        
        acc_index = np.zeros(num_cls).astype(np.int16)
        for i in range(num_cls):
            acc_index[i] = sum(train_per_class[:i+1])
        
        centers = ck[1]
        for i in range(1, num_cls):
            centers = torch.cat([centers, ck[i+1]], dim=1)
            
        radius = torch.norm(centers, p=2, dim=0).mean()
        centers_trans = centers / torch.torch.norm(centers, p=2, dim=0) * radius
        
        result = torch.zeros(num_train).cuda()
        for i in range(num_train):
            class_ind = bisect.bisect(acc_index, i)
            #print(class_ind)
            if class_ind == 0:
                negative_centers = centers[:, 1:]
            else:
                negative_centers = torch.cat([centers[:, :class_ind], centers[:, class_ind+1:]], dim=1)
            #print(negative_centers.size())
            closest_negative_center_index = torch.min(torch.norm(feat[:, i].unsqueeze(dim=1) - negative_centers, 
                                                                 p=2, dim=0), dim=0)[1]
            closest_negative_center = negative_centers[:, closest_negative_center_index]
            closest_negative_center_trans = closest_negative_center / torch.norm(closest_negative_center, p=2) * radius
            interclass_dists = torch.square(torch.dist(centers_trans[:, class_ind], closest_negative_center_trans, p=2))
            #print(centers_dists)
            intraclass_dists = torch.square(torch.dist(feat[:, i], centers_trans[:, class_ind], p=2))
            result[i] =  intraclass_dists - interclass_dists * tan_beta_2 + radius ** 2 / 2
            if result[i] < 0.0:
                result[i] = 0.0
            #print(result[i])
        #print(result.sum() / (2*num_train))
        return result.sum() / (2 * num_train)

    def statistical_loss(self, y, train_inds, final_feat):  
        #y: (1, 624, 352), train_inds: (1, 624, 352), final_feat: (1, 103, 624, 352)
        lamb = 1e-3
        y = y.squeeze()  # (624, 352)
        train_inds = train_inds.squeeze()  #(624, 352)
        final_feat = final_feat.squeeze()  # (103, 624, 352)
        
        cls_list = torch.unique(y)  # tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
        num_cls = len(cls_list) - 1
        num_train = int(train_inds.sum())
        feat_dimension = final_feat.size()[0]  # 103
        
        location = torch.where(train_inds == 1.)
        label = y[location]
        
        feat_dict_per_class = dict()
        for i in range(1, num_cls+1):
            feat_inds = torch.where(label==i)
            feat_dict_per_class[i] = final_feat[:, location[0][feat_inds], location[1][feat_inds]]
            
        ck = dict()
        for i in range(1, num_cls+1):
            ck[i] = feat_dict_per_class[i].mean(dim=1).unsqueeze(dim=1)
            
        variance_loss = torch.tensor(0.).cuda()
        for i in range(1, num_cls+1):
            zj_ck = feat_dict_per_class[i] - ck[i]  # (103, num_train)
            variance_loss += zj_ck.mul(zj_ck).sum() / (zj_ck.size()[1] - 1)
        variance_loss = variance_loss / num_cls
        
        diver_loss = torch.tensor(0.).cuda()
        for k in range(1, num_cls+1):
            for t in range(k+1, num_cls+1):
                diver_loss -= torch.dist(ck[k], ck[t], p=2)
        diver_loss = diver_loss * lamb
        #print(variance_loss, diver_loss)
        return variance_loss + diver_loss
        
    def set_defalut_config(self):
        self.config.update(dict(
            in_channels=103,
            num_classes=9,
            block_channels=(96, 128, 192, 256),
            num_blocks=(1, 1, 1, 1),
            inner_dim=128,
            reduction_ratio=1.0,
        ))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        y = x * out.view(out.size(0), out.size(1), 1, 1)
        y = y + residual
        return y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out)
        out2 = self.relu1(out1)
        out = self.sigmoid(out2)
        y = x * out.view(out.size(0), 1, out.size(-2), out.size(-1))
        y = y + residual
        return y


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.ca(x), self.sa(x)], dim=1)
        return out


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = tuple((int((i-1)/2) for i in kernel_size))

        self.Wxi = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv3d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv3d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).cuda()
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).cuda()
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1], shape[2])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Dimension Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
            assert shape[2] == self.Wci.size()[4], 'Input Height Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1], shape[2])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1], shape[2])).cuda())

    
class Si_ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=8, effective_step=7):
        super(Si_ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)
    
    def forward(self, input):
        internal_state = []
        outputs = []
        a = input.size(1)
        b = int(a / 8)
        for step in range(self.step):
            x = input[:, step * b:(step + 1) * b, :, :]  # (1, 12, 624, 352), (1, 16, 312, 176)...
            x = x.unsqueeze(dim=1)  # (1, 1, 12, 624, 352), (1, 1, 16, 312, 176)...
            
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = "cell{}".format(i)
                if step == 0:
                    bsize, _, dimension, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], 
                                                             shape=(dimension, height, width))
                    internal_state.append((h, c))
                
                # do forward 
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
        # only record last steps
        result = x[:, 0]
        for i in range(self.hidden_channels[-1] - 1):
            result = torch.cat([result, x[:, i + 1]], dim=1)
        return result  # (batch_size, self.hidden_channels[-1] * 8, height, width)
    

class Bi_ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, step=8, effective_step=7):
        super(Bi_ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            name_reverse = 'cell_reverse{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            cell_reverse = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            setattr(self, name_reverse, cell_reverse)
            self._all_layers.append(cell)
            self._all_layers.append(cell_reverse)

    def forward(self, input):
        internal_state = []
        internal_state_reverse = []
        outputs = []
        outputs_reverse = []
        a = input.squeeze()
        b = int(len(a) / 8)
        for i in range(self.num_layers):
            name = "cell{}".format(i)
            name_reverse = "cell_reverse{}".format(i)
            if i == 0:
                for step in range(self.step):
                    if step == 0:
                        x_reverse = input[:, -(step + 1) * b:]
                    else:
                        x_reverse = input[:, -(step + 1) * b: -step * b]
                    x_reverse = x_reverse.unsqueeze(dim=1)  # (1, 1, 12, 624, 352), (1, 1, 16, 312, 176)...
                    x = input[:, step * b:(step + 1) * b, :, :]  # (1, 12, 624, 352), (1, 16, 312, 176)...
                    x = x.unsqueeze(dim=1)  # (1, 1, 12, 624, 352), (1, 1, 16, 312, 176)...
                    bsize, _, dimension, height, width = x.size()
                    if step == 0:
                        (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], 
                                                                 shape=(dimension, height, width))
                        internal_state.append((h, c))
                        (h_reverse, c_reverse) = getattr(self, name_reverse).init_hidden(batch_size=bsize, 
                                                                                         hidden=self.hidden_channels[i], 
                                                                                         shape=(dimension, height, width))
                        internal_state_reverse.append((h_reverse, c_reverse))
                    # do forward
                    (h, c) = internal_state[i]
                    (h_reverse, c_reverse) = internal_state_reverse[i]
                    x, new_c = getattr(self, name)(x, h, c)
                    internal_state[i] = (x, new_c)
                    x_reverse, new_c_reverse = getattr(self, name_reverse)(x_reverse, h_reverse, c_reverse)
                    internal_state_reverse[i] = (x_reverse, new_c_reverse)
                    outputs.append(x)
                    outputs_reverse.insert(0, x_reverse)
                if self.num_layers == 1:
                    last_step_output = outputs[-1] + outputs_reverse[-1]
                    result = last_step_output[:, 0]
                    for j in range(self.hidden_channels[i] - 1):
                        result = torch.cat([result, last_step_output[:, j + 1]], dim=1)
                    return result
            else:
                input = torch.cat([outputs[j] + outputs_reverse[j] for j in range(self.step)], dim=1)
                b = self.hidden_channels[i - 1]
                outputs = []
                outputs_reverse = []
                for step in range(self.step):
                    if step == 0:
                        x_reverse = input[:, -(step + 1) * b:]
                    else:
                        x_reverse = input[:, -(step + 1) * b: -step * b]
                    x = input[:, step * b:(step + 1) * b]  # (1, 8, 12, 624, 352), (1, 8, 16, 312, 176)...
                    bsize, _, dimension, height, width = x.size()
                    if step == 0:
                        (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i], 
                                                                 shape=(dimension, height, width))
                        internal_state.append((h, c))
                    
                        (h_reverse, c_reverse) = getattr(self, name_reverse).init_hidden(batch_size=bsize, 
                                                                                         hidden=self.hidden_channels[i], 
                                                                                         shape=(dimension, height, width))
                        internal_state_reverse.append((h_reverse, c_reverse))
                    # do forward
                    (h, c) = internal_state[i]
                    (h_reverse, c_reverse) = internal_state_reverse[i]
                    x, new_c = getattr(self, name)(x, h, c)
                    internal_state[i] = (x, new_c)
                    x_reverse, new_c_reverse = getattr(self, name_reverse)(x_reverse, h_reverse, c_reverse)
                    internal_state_reverse[i] = (x_reverse, new_c_reverse)
                    outputs.append(x)
                    outputs_reverse.insert(0, x_reverse)
                if i == self.num_layers - 1:
                    last_step_output = outputs[-1] + outputs_reverse[-1]
                    result = last_step_output[:, 0]
                    for j in range(self.hidden_channels[i] - 1):
                        result = torch.cat([result, last_step_output[:, j + 1]], dim=1)
                    return result
