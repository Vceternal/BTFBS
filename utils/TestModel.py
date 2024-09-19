# -*- coding:utf-8 -*-


import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve, roc_curve,
                             precision_score, recall_score, roc_auc_score, average_precision_score)


def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for data in pbar:
            '''data preparation '''
            compounds, proteins, labels = data
            compounds = compounds.to(DEVICE)
            proteins = proteins.to(DEVICE)
            labels = labels.to(DEVICE)

            if isinstance(MODEL, list):
                predicted_scores = torch.zeros(2).to(DEVICE)
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + \
                        MODEL[i](compounds, proteins)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores = MODEL(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(
                predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())
    test_num = len(Y)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if Y[index] == 1:
            if Y[index] == P[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if  Y[index] == P[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    ACC = float(tp + tn) / test_num
        
    if tp + fn == 0:
        Sensitivity = 0
    else:
        Sensitivity = float(tp) / (tp + fn)
    if tn + fp == 0:
        Specificity=0
    else:
        Specificity = float(tn) / (tn + fp) 
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))                                                    
    Precision = precision_score(Y, P)
    Reacll = recall_score(Y, P)
    if Reacll + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Reacll * Precision / (Reacll + Precision)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    prc = average_precision_score(Y, S)
    fpr1, tpr1, thr= roc_curve(Y, S)
    roc_auc= auc(fpr1, tpr1)

    test_loss = np.average(test_losses)
    return Y, P, test_loss, F1, ACC, Sensitivity, Specificity, MCC, Reacll,Precision


def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1):
    #test_pbar = tqdm(
        #enumerate(
            #BackgroundGenerator(dataset_loader)),
        #total=len(dataset_loader))
    T, P, loss_test, F1_test, ACC_test, Sensitivity_test, Specificity_test, MCC_test, Recall_test, precision_test= test_precess(
        MODEL, dataset_loader, LOSS, DEVICE, FOLD_NUM)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                "/{}_{}_prediction.txt".format(DATASET, dataset_class)
        else:
            filepath = save_path + \
                "/{}_{}_ensemble_prediction.txt".format(DATASET, dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + '\n')
        with open('MCC.txt', 'a') as f:
            
            f.write('真实：'+str(T))
            f.write('\n')
            # f.write('---------------------------------------------------------------------------------------------')
            f.write('预测：'+str(P))   
            f.write('\n')    
    results = '{}: Loss:{:.5f};F1:{:.5f};ACC:{:.5f};Sensitivity:{:.5f};Specificity:{:.5f};MCC:{:.5f};Recall:{:.5f}.' \
        .format(dataset_class, loss_test, F1_test, ACC_test, Sensitivity_test, Specificity_test, MCC_test,Recall_test)
    print(results)
    return results, F1_test, ACC_test, Sensitivity_test, Specificity_test, MCC_test,Recall_test,precision_test
