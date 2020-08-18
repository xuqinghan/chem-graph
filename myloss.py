import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as TF
import torchvision.utils as vutils
import torch.nn.functional as F
 
class HingeLoss(nn.Module):
    """
    铰链损失
    SVM hinge loss
    L1 loss = sum(max(0,pred-true+1)) / batch_size
    注意： 此处不包含正则化项, 需要另外计算出来 add
    https://blog.csdn.net/AI_focus/article/details/78339234
    """
 
    def __init__(self, reduction='sum'):
        super(HingeLoss, self).__init__()

 
    def forward(self, y_pred, y_truth):
        # y_pred: [b,n]
        # y_truth:[b,n]
        batch_size = y_truth.size(0)
        dim_y = y_truth.size(1)
        #print('y_pred', y_pred)
        #print('y_truth', y_truth)
        #print(batch_size, dim_y)
        ones = torch.ones(batch_size, dim_y, dtype=torch.float).cuda()
        loss = ones - torch.mul(y_pred, y_truth)
        #print(loss)
        loss[loss < 0] = 0
        #print(loss)
        #每批返回1个结果
        return torch.sum(loss) / batch_size
