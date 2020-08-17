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
 
    def __init__(self, n_classes, margin=1.):
        super(HingeLoss, self).__init__()
        self.margin = margin
        self.n_classes = n_classes
 
    def forward(self, y_pred, y_truth):
        # y_pred: [b,n]
        # y_truth:[b,n]
        batch_size = y_truth.size(0)
        mask = torch.eye(self.n_classes, self.n_classes, dtype=torch.bool)[y_truth].cuda()
        y_pred_true = torch.masked_select(y_pred, mask).unsqueeze(dim=-1).cuda()
        loss = torch.max(torch.zeros_like(y_pred).cuda(), y_pred - y_pred_true + self.margin)
        loss = loss.masked_fill(mask, 0)
        return torch.sum(loss) / batch_size