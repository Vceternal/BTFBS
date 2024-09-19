# -*- coding:utf-8 -*-


import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from config import hyperparameter
from model import BTFBS
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.EarlyStoping import EarlyStopping
from LossFunction import CELoss, PolyLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_model(SEED, DATASET, MODEL, K_Fold, LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''init hyperparameters'''
    hp = hyperparameter()

    '''load dataset from text file'''
    assert DATASET in [ "Bdata_RS","Bdata_EE"]
    print("Train in " + DATASET)
    print("load data")
    dir_input = ('./DataSets/{}.txt'.format(DATASET))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    print("load finished")

    '''set loss function weight'''

    weight_loss = None

    '''shuffle data'''
    print("data shuffle")
    data_list = shuffle_dataset(data_list, SEED)
    

    '''split dataset to train&validation set and test set'''
    split_pos = len(data_list) - int(len(data_list) * 0.2)
    train_data_list = data_list[0:split_pos]
    test_data_list = data_list[split_pos:-1]
    print(test_data_list)
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))

    '''metrics'''
    F1_List_stable, ACC_List_stable, Sensitivity_List_stable, Specificity_List_stable, MCC_List_stable, Recall_List_stable, Precison_List_stable = [], [], [], [], [], [], []

    for i_fold in range(K_Fold):

        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_kfold_data(
            i_fold, train_data_list, k=K_Fold)
        train_dataset = CustomDataSet(train_dataset)
        valid_dataset = CustomDataSet(valid_dataset)
        test_dataset = CustomDataSet(test_data_list)
        train_size = len(train_dataset)

        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                          collate_fn=collate_fn, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                          collate_fn=collate_fn, drop_last=True)
        test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                         collate_fn=collate_fn, drop_last=True)

        """ create model"""
        model = MODEL(hp).to(DEVICE)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]

        """create optimizer and scheduler"""
        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        if LOSS == 'PolyLoss':
            Loss = PolyLoss(weight_loss=weight_loss,
                            DEVICE=DEVICE, epsilon=hp.loss_epsilon)
        else:
            Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

        """Output files"""
        save_path = "./" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')

        for epoch in range(1, hp.Epoch + 1):
            

            if early_stopping.early_stop == True:
                break
            
            
            """train"""
            train_losses_in_epoch = []
            model.train()
            y,p1,s=[],[],[]
            for train_data in train_dataset_loader:
                trian_compounds, trian_proteins, trian_labels = train_data
                trian_compounds = trian_compounds.to(DEVICE)
                trian_proteins = trian_proteins.to(DEVICE)
                trian_labels = trian_labels.to(DEVICE)

                optimizer.zero_grad()

                predicted_interaction = model(trian_compounds, trian_proteins)
                train_loss = Loss(predicted_interaction, trian_labels)
                train_labels = trian_labels.to('cpu').data.numpy()
                train_scores = F.softmax(
                        predicted_interaction, 1).to('cpu').data.numpy()
                    
                train_predictions = np.argmax(train_scores, axis=1)
                y.extend(train_labels)
                p1.extend(train_predictions)
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  # 一次epoch的平均训练loss
            

            test_num = len(y)
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for index in range(test_num):
                if y[index] == 1:
                    if y[index] == p1[index]:
                        tp = tp + 1
                    else:
                        fn = fn + 1
                else:
                    if  y[index] == p1[index]:
                        tn = tn + 1
                    else:
                        fp = fp + 1
            ACC = float(tp + tn) / test_num

            """valid"""
           
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_data in valid_dataset_loader:

                    valid_compounds, valid_proteins, valid_labels = valid_data

                    valid_compounds = valid_compounds.to(DEVICE)
                    valid_proteins = valid_proteins.to(DEVICE)
                    valid_labels = valid_labels.to(DEVICE)

                    valid_scores = model(valid_compounds, valid_proteins)
                    
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(
                        valid_scores, 1).to('cpu').data.numpy()
                    
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)
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
            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            if Reacll_dev + Precision_dev == 0:
                F1 = 0
            else:
                F1 = 2 * Reacll_dev * Precision_dev / (Reacll_dev + Precision_dev) 
            Accuracy_dev = accuracy_score(Y, P)
            AUC_dev = roc_auc_score(Y, S)
            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)
           

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_Sensitivity: {Sensitivity:.5f} ' +
                         f'valid_Specificity: {Specificity:.5f} ' +
                         f'valid_F1: {F1:.5f} ' +
                         f'valid_ACC: {ACC:.5f} ' +
                         f'valid_MCC: {MCC:.5f} '+
                         f'valid_Precision: {Precision_dev:.5f} '
                         )
            print(print_msg)
        
            '''save checkpoint and make decision when early stop'''
            early_stopping(Accuracy_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(
            early_stopping.savepath + '/valid_best_checkpoint.pth'))

        '''test model'''
        trainset_test_stable_results, _, _, _, _, _,_,_ = test_model(
            model, train_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Train", FOLD_NUM=1)
        validset_test_stable_results, _, _, _, _, _,_,_ = test_model(
            model, valid_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Valid", FOLD_NUM=1)
        testset_test_stable_results, F1_test, ACC_test, Sensitivity_test, Specificity_test, MCC_test,Recall_test,precision_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, DEVICE, dataset_class="Test", FOLD_NUM=1)
        ACC_List_stable.append(ACC_test)
        F1_List_stable.append(F1_test)
        Sensitivity_List_stable.append(Sensitivity_test)
        Specificity_List_stable.append(Specificity_test)
        MCC_List_stable.append(MCC_test)
        Recall_List_stable.append(Recall_test)
        Precison_List_stable.append(precision_test)
        with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
            f.write("Test the stable model" + '\n')
            f.write(trainset_test_stable_results + '\n')
            f.write(validset_test_stable_results + '\n')
            f.write(testset_test_stable_results + '\n')

    show_result(DATASET, F1_List_stable, ACC_List_stable,
                Sensitivity_List_stable, Specificity_List_stable, MCC_List_stable,Recall_List_stable, Precison_List_stable, Ensemble=False)



