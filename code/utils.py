import math
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


# Evaluation metrics: Acc, Auroc, Auprc

def overall_acc(prob, gt):
    prob = prob.cpu().data.numpy().flatten()
    prediction = [round(a) for a in prob]
    gt = gt.cpu().data.numpy().flatten()
    acc = accuracy_score(gt, prediction)
    return acc

def auroc(prob, gt):
    y_true = gt.cpu().data.numpy().flatten()
    y_scores = prob.cpu().data.numpy().flatten()
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auroc_score = auc(fpr, tpr)
    return auroc_score

def auprc(prob, gt):
    y_true = gt.cpu().data.numpy().flatten()
    y_scores = prob.cpu().data.numpy().flatten()
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    auprc_score = auc(recall, precision)
    return auprc_score