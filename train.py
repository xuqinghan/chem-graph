from time import process_time

import torch
import torch.nn.functional as F
from torch import nn
#tensorboard --logdir=logs
from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='logs',comment='train-loss')

from models import Net
from prepare_data import loader_tr, loader_ts

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = Net(50, 500, 2500)
criterion = nn.HingeEmbeddingLoss()
lr = 0.05
momentum = 0.9
#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

#criterion = nn.MSELoss()
#criterion = nn.L1Loss()


model.to(device)
losses_tr = []
losses_ts = []


num_epoches = 100
#训练

for epoch in range(num_epoches):
    print(f'epoch {epoch}, training....')
    loss_tr = 0
    acc_tr = 0
    if epoch %5 == 0:
        #每隔5轮
        optimizer.param_groups[0]['lr'] *=0.9
    
    optimizer.zero_grad()    
    model.train()
    start = process_time()
    for data in loader_tr:

        data = data.to(device)
        out = model(data)
        #print(out.shape)
        #print(data.y.shape)
        loss =criterion(out, data.y)

        loss.backward()
        optimizer.step()
        #记录误差 一个数
        loss_tr += loss.item()
    

    #累计当前批次的损失
    loss_tr_epoch = loss_tr/len(loader_tr)
    print(f'train done. {process_time()- start} sec, loss={loss_tr_epoch:.4f}')
    losses_tr.append(loss_tr_epoch)
    # 保存loss的数据与epoch数值
    writer.add_scalar('Train', loss_tr_epoch, epoch)



    loss_ts = 0
    #改成eval模式
    model.eval()
    for data in loader_ts:
        #x y 送入 gpu
        data = data.to(device)

        out = model(data)
        #print(out.shape)
        #print(data.y.shape)
        loss =criterion(out, data.y)

        #记录误差 一个数
        loss_ts += loss.item()

    #累计当前批次的损失
    loss_ts_epoch = loss_ts/len(loader_ts)
    print(f'test sec, loss={loss_ts_epoch:.4f}')
    losses_ts.append(loss_ts_epoch)
    writer.add_scalar('Test', loss_ts_epoch, epoch)

#保存模型
torch.save(model, 'model.pkl')