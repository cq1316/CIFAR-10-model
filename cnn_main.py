# -*- coding: utf-8 -*
import math
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
from DesignatedCNN.utils.get_set import get_data_set
from DesignatedCNN.models.cifarmodel import CifarModel
import sys
import logging
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


def train_batch(epoch, train_set, optimizer, model, batchsize):
    batch_num = math.ceil(len(train_set) / batchsize)
    train_loss = 0
    model.train(mode=True)
    for n in range(0, batch_num):
        targets = []
        lesion_imgs = []
        for img_ki67 in train_set[n * batchsize:(n + 1) * batchsize]:
            lesion_img = Variable(img_ki67.get_tensor().unsqueeze(0))
            lesion_imgs.append(lesion_img)
            target = img_ki67.get_class()
            targets.append(target)
        lesion_imgs = torch.cat(lesion_imgs).cuda()
        targets = Variable(torch.LongTensor(targets)).cuda()
        optimizer.zero_grad()
        output = model(lesion_imgs)
        loss = F.cross_entropy(input=output,
                               target=targets,
                               weight=torch.FloatTensor([0.75, 0.25]).cuda())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_set)


def test(test_set, model, print_id=False):
    y_t = []
    y_p = []
    model.eval()
    for img_ki67 in test_set:
        # 取出mrmr特征
        lesion_img = Variable(img_ki67.get_tensor().unsqueeze(0)).cuda()
        target = img_ki67.get_class()
        y_t.append(target)
        output = model(lesion_img)
        y_p.append(output.data[0].cpu()[1])
    return y_t, y_p


def evaluate_on_trainset(y_t, y_p):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for t, p in zip(y_t, y_p):
        if p >= 0.5:
            if t == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        else:
            if t == 1:
                FN = FN + 1
            else:
                TN = TN + 1
    acc = (TP + TN) / (FP + FN + TP + TN)
    return acc, TP, TN, FP, FN


if __name__ == "__main__":
    path = sys.argv[1]
    logging.info("get data set")
    # get_data_set 正确
    train_neg_set, train_pos_set, test_pos_set, test_neg_set = get_data_set(path=path,
                                                                            resize_length=32)
    total_train_set = train_neg_set + train_pos_set
    total_test_set = test_neg_set + test_pos_set
    length = len(train_neg_set)
    train_set = train_pos_set + train_neg_set
    logging.info("get data set completed")
    for i in range(30):
        logging.info("train %d experiment" % i)
        lr = 0.001
        model = CifarModel().cuda()
        classifier_optimizer = optim.Adam(model.parameters(),
                                          lr=lr,
                                          weight_decay=5e-4)
        runs = 20
        accs = [0 for j in range(runs)]
        pre_acc = 0
        errors = [100 for j in range(runs)]
        pre_error = 0
        change_lr = False
        for epoch in range(1, 3000):
            if change_lr:
                for param_group in classifier_optimizer.param_groups:
                    param_group['lr'] = lr
            random.shuffle(train_set)
            error = train_batch(epoch=epoch,
                                train_set=train_set,
                                optimizer=classifier_optimizer,
                                model=model,
                                batchsize=16)
            # logging.info("Testing trainset")
            y_t1, y_p1 = test(test_set=total_train_set, model=model)
            fpr_tr, tpr_tr, threshold = roc_curve(y_t1, y_p1, pos_label=1)
            roc_auc_tr = auc(fpr_tr, tpr_tr)
            # logging.info("auc: %f" % roc_auc_tr)
            acc_tr, TP_tr, TN_tr, FP_tr, FN_tr = evaluate_on_trainset(y_t1, y_p1)
            # logging.info("Testing testset")
            y_t2, y_p2 = test(test_set=total_test_set, model=model)
            fpr_te, tpr_te, threshold = roc_curve(y_t2, y_p2, pos_label=1)
            roc_auc_te = auc(fpr_te, tpr_te)
            # logging.info("auc: %f" % roc_auc_te)
            acc_te, TP_te, TN_te, FP_te, FN_te = evaluate_on_trainset(y_t2, y_p2)
            # lr变化条件
            error_mean = sum(errors) / runs
            if abs(error - pre_error) < 0.001 and abs(error - error_mean) < 0.001:
                lr = lr * 0.8
                change_lr = True
                # logging.info("===================================")
                # logging.info("lr : %s" % (str(lr)))
                errors = [100 for j in range(runs)]
            else:
                change_lr = False
                errors.pop(0)
                errors.append(error)
                pre_error = error
            acc_mean = sum(accs) / runs
            if round(acc_tr, ndigits=3) == round(acc_mean, ndigits=3) and \
                            round(acc_tr, ndigits=3) == round(pre_acc, ndigits=3) and epoch > 50:
                cols = ["train_tp", "train_tn", "train_fp", "train_fn", "train_acc", "train_auc",
                        "test_tp", "test_tn", "test_fp", "test_fn", "test_acc", "test_auc",
                        "train_fpr", "train_tpr",
                        "test_fpr", "test_tpr"]
                data = [TP_tr, TN_tr, FP_tr, FN_tr, acc_tr, roc_auc_tr,
                        TP_te, TN_te, FP_te, FN_te, acc_te, roc_auc_te,
                        fpr_tr, tpr_tr, fpr_te, tpr_te]
                experiment_df = pd.DataFrame(data=[data], columns=cols)
                experiment_df.to_csv("experiment_%d.csv" % i, index=0)
                break
            else:
                accs.pop(0)
                accs.append(acc_tr)
                pre_acc = acc_tr
