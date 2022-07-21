import numpy as np
from numpy import *
import sklearn
import torch

from model import DiMiMeNet
from utils import overall_acc, auroc, auprc


# Load Datasets

# node feature
di_feature = np.load('Dataset/di_feature.npy', allow_pickle=True)
me_feature = np.load('Dataset/me_feature.npy', allow_pickle=True)
# edge data
di_mi = np.load('Dataset/di_mi.npy', allow_pickle=True)
me_mi = np.load('Dataset/me_mi.npy', allow_pickle=True)
mi_mi = np.load('Dataset/mi_mi.npy', allow_pickle=True)
# positive and negative samples
di_me_p = np.load('Dataset/di_me_p.npy', allow_pickle=True)
di_me_n = np.load('Dataset/di_me_n_random1.npy', allow_pickle=True)

label = np.array([1]*di_me_p.shape[1]+[0]*di_me_n.shape[1])
di_me = np.hstack((di_me_p,di_me_n))
di_me = di_me[[1,0],:]


# Spliting the data set into training set, verification set and test set.

Idx = range(len(di_me[0]))
idx = sklearn.utils.shuffle(Idx, random_state=1)
train_id = idx[:int(0.8*len(idx))]
val_id = idx[int(0.8*len(idx)):int(0.9*len(idx))]
test_id = idx[int(0.9*len(idx)):]

di_me_train, di_me_val, di_me_test = di_me[:, train_id], di_me[:, val_id], di_me[:, test_id]
label_train, label_val, label_test = label[train_id], label[val_id], label[test_id]


# Define model parameters

hidden_dim_1 = 512
hidden_dim_2 = 128

batch_num = 512

global_mi_num = 965
global_me_num = 2511
global_di_num = 56

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if USE_CUDA else 'cpu')


# Select disease and metabolite feature

select_me = np.unique(di_me_train[0])
select_di = np.unique(di_me_train[1])
di_feature = di_feature[:, select_di]
me_feature = me_feature[:, select_me]


# Load the data into tensor

di_feature = torch.tensor(list(di_feature), dtype=torch.float).float().to(device)
me_feature = torch.tensor(list(me_feature), dtype=torch.float).float().to(device)
mi_feature = torch.zeros(global_mi_num, hidden_dim_1).float().to(device)

mi_mi = torch.from_numpy(mi_mi).long().to(device)
di_mi = torch.from_numpy(di_mi).long().to(device)
me_mi = torch.from_numpy(me_mi).long().to(device)


# Load model and define hyper-parameters

model = DiMiMeNet(hidden_dim_1, hidden_dim_2,
                        global_mi_num, global_me_num, global_di_num,
                        mi_feature_num=hidden_dim_1, me_feature_num=len(select_me),
                        di_feature_num=len(select_di)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)


# Functions of model training, verification and testing

def train_val_test(phase, epoch, batch, model, mi_mi, me_mi, di_mi,
                   me_feature, di_feature, mi_feature, pair, gt):
    if phase == 'Train':
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    logging_info = {}
    logging_info.update({'%s Epoch' % phase: epoch, 'batch': batch})
    prob = model(mi_mi, me_mi, di_mi, me_feature, di_feature, mi_feature, pair)
    gt = gt.float().to(device)

    weight = class_weight[gt.long()].to(device)
    loss_func = torch.nn.BCELoss(weight=weight, reduction='mean').to(device)
    loss = loss_func(prob, gt)
        
    if phase == 'Train':
        loss.backward()
        optimizer.step()

    logging_info.update({'loss': '%.04f' % loss.data.item()})
    logging_info.update({'auroc': '%.04f' % auroc(prob, gt)})
    logging_info.update({'auprc': '%.04f' % auprc(prob, gt)})
    logging_info.update({'acc': '%.04f' % overall_acc(prob, gt)})
    return logging_info


# Calculation
# initialization
class_weight = torch.Tensor([1, 1])
train_loss = []
train_acc = []
train_auroc = []
train_auprc = []
val_loss = []
val_acc = []
val_auroc = []
val_auprc = []
test_loss = []
test_acc = []
test_auroc = []
test_auprc = []
best_metric = 0.5
best_epoch = -1

# 500 epochs for training the model
for epoch in range(500):
    batch_loss = []
    batch_acc = []
    batch_auroc = []
    batch_auprc = []
    for batch, idx in enumerate(torch.split(torch.randperm(len(train_id)), batch_num)):
        train_logging = train_val_test('Train', epoch, batch, model, mi_mi, me_mi, di_mi,
                                        me_feature, di_feature, mi_feature,
                                        pair=di_me_train[:, idx],
                                        gt=torch.from_numpy(label_train[idx]))
        batch_loss.append(float(train_logging['loss']))
        batch_acc.append(float(train_logging['acc']))
        batch_auroc.append(float(train_logging['auroc']))
        batch_auprc.append(float(train_logging['auprc']))
    train_loss.append(mean(batch_loss))
    train_acc.append(mean(batch_acc))
    train_auroc.append(mean(batch_auroc))
    train_auprc.append(mean(batch_auprc))
    print('Epoch : ',epoch+1,' | ','Train loss: ',format(mean(batch_loss), '.4f'),' | ','Train acc: ',format(mean(batch_acc), '.4f'),' | ','Train auroc: ',format(mean(batch_auroc), '.4f'),' | ','Train auprc: ',format(mean(batch_auprc), '.4f'))
    
    batch_loss = []
    batch_acc = []
    batch_auroc = []
    batch_auprc = []
    for batch, idx in enumerate(torch.split(torch.randperm(len(val_id)), batch_num)):
        val_logging = train_val_test('Val', epoch, batch, model, mi_mi, me_mi, di_mi,
                                        me_feature, di_feature, mi_feature,
                                        pair=di_me_val[:, idx],
                                        gt=torch.from_numpy(label_val[idx]))
        batch_loss.append(float(val_logging['loss']))
        batch_acc.append(float(val_logging['acc']))
        batch_auroc.append(float(val_logging['auroc']))
        batch_auprc.append(float(val_logging['auprc']))
    val_loss.append(mean(batch_loss))
    val_acc.append(mean(batch_acc))
    val_auroc.append(mean(batch_auroc))
    val_auprc.append(mean(batch_auprc))
    print('Epoch : ',epoch+1,' | ','Val loss: ',format(mean(batch_loss), '.4f'),' | ','Val acc: ',format(mean(batch_acc), '.4f'),' | ','Val auroc: ',format(mean(batch_auroc), '.4f'),' | ','Val auprc: ',format(mean(batch_auprc), '.4f'))
    
    # select the best model based on metrics of the verification set
    if mean(batch_auroc) > best_metric:
        best_metric = mean(batch_auroc)
        best_epoch = epoch+1
        torch.save(model.state_dict(), 'DiMiMe.model')
    print('Best Epoch: ', best_epoch, ' | Best Auroc: ', format(best_metric, '.4f'))
    
    batch_loss = []
    batch_acc = []
    batch_auroc = []
    batch_auprc = []
    for batch, idx in enumerate(torch.split(torch.randperm(len(test_id)), batch_num)):
        test_logging = train_val_test('Test', epoch, batch, model, mi_mi, me_mi, di_mi,
                                    me_feature, di_feature, mi_feature,
                                    pair=di_me_test[:, idx],
                                    gt=torch.from_numpy(label_test[idx]))
        batch_loss.append(float(test_logging['loss']))
        batch_acc.append(float(test_logging['acc']))
        batch_auroc.append(float(test_logging['auroc']))
        batch_auprc.append(float(test_logging['auprc']))
    test_loss.append(mean(batch_loss))
    test_acc.append(mean(batch_acc))
    test_auroc.append(mean(batch_auroc))
    test_auprc.append(mean(batch_auprc))
    print('Epoch : ',epoch+1,' | ','Test loss: ',format(mean(batch_loss), '.4f'),' | ','Test acc: ',format(mean(batch_acc), '.4f'),' | ','Test auroc: ',format(mean(batch_auroc), '.4f'),' | ','Test auprc: ',format(mean(batch_auprc), '.4f'))

    scheduler.step()