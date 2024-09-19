# -*- coding:utf-8 -*-


import numpy as np


def show_result(DATASET, F1_List, ACC_List, Sensitivity_List, Specificity_List, MCC_List, Recall_List,Precision_List, Ensemble=False):
    F1_mean, F1_var = np.mean(F1_List), np.var(F1_List)
    ACC_mean, ACC_var = np.mean(
        ACC_List), np.var(ACC_List)
    Sensitivity_mean, Sensitivity_var = np.mean(Sensitivity_List), np.var(Sensitivity_List)
    Specificity_mean, Specificity_var = np.mean(Specificity_List), np.var(Specificity_List)
    MCC_mean, MCC_var = np.mean(MCC_List), np.var(MCC_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    Precision_mean, Precision_var = np.mean( Precision_List), np.var( Precision_List)

    if Ensemble == False:
        print("The model's results:")
        filepath = "./{}/results.txt".format(DATASET)
    else:
        print("The ensemble model's results:")
        filepath = "./{}/ensemble_results.txt".format(DATASET)
    with open(filepath, 'w') as f:
        f.write('F1(std):{:.4f}({:.4f})'.format(
            F1_mean, F1_var) + '\n')
        f.write('ACC(std):{:.4f}({:.4f})'.format(
            ACC_mean, ACC_var) + '\n')
        f.write('Sensitivity(std):{:.4f}({:.4f})'.format(
            Sensitivity_mean, Sensitivity_var) + '\n')
        f.write('Specificity(std):{:.4f}({:.4f})'.format(Specificity_mean, Specificity_var) + '\n')
        f.write('MCC(std):{:.4f}({:.4f})'.format(MCC_mean, MCC_var) + '\n')
    print('F1(std):{:.4f}({:.4f})'.format(F1_mean, F1_var))
    print('ACC(std):{:.4f}({:.4f})'.format(
        ACC_mean, ACC_var))
    print('Sensitivity(std):{:.4f}({:.4f})'.format(Sensitivity_mean, Sensitivity_var))
    print('Specificity(std):{:.4f}({:.4f})'.format(Specificity_mean, Specificity_var))
    print('MCC(std):{:.4f}({:.4f})'.format(MCC_mean, MCC_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))

