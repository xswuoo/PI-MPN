import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    a = torch.mean(loss)
    return torch.mean(loss)


def MSE(preds, labels, null_val=np.nan):
    # labels[labels < 5.01] = 0.0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def RMSE(preds, labels, null_val=np.nan):
    # labels[labels < 5.01] = 0.0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sqrt(torch.mean(loss))


def MAPE(preds,labels, null_val=np.nan):
    # labels[labels < 5.01] = 0.0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels>10.0)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    a = torch.mean(loss)
    b = torch.mean(loss).item()
    return torch.mean(loss)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def r_squared(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    preds = preds[mask]
    labels = labels[mask]
    mean_labels = torch.mean(labels)
    tss = torch.sum((labels - mean_labels) ** 2)
    rss = torch.sum((labels - preds) ** 2)
    r_squared = 1 - (rss / tss)
    return r_squared

def explained_variance_score(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    var_error = torch.var(preds - labels)
    var_labels = torch.var(labels)
    ev = 1 - (var_error / var_labels)
    loss = ev * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, true):
    pred = torch.from_numpy(pred)
    true = torch.from_numpy(true)
    mae = MAE(pred, true).item()
    mse = MSE(pred, true).item()
    rmse = RMSE(pred, true).item()
    mape = MAPE(pred, true, 0.0).item()
    r_2 = r_squared(pred, true).item()
    return mae, mse, rmse, mape, r_2
