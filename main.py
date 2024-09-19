# -*- coding:utf-8 -*-


import argparse
import os
from RunModel import run_model
from model import BTFBS
#os.environ["CUDA_VISIBLE_DEVICES"] = "X"


parser = argparse.ArgumentParser(
    prog='BTFBS',
    description='BTFBS is model in paper: \"BTFBS\"',
    epilog='Model config set by config.py')

parser.add_argument('dataSetName', choices=[
                    "Bdata_RS", "Bdata_EE"], help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['BTFBS'],
                    default='BTFBS', help='Which model to use, \"BTFBS\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')
args = parser.parse_args()

if args.model == 'BTFBS':
    run_model(SEED=args.seed, DATASET=args.dataSetName,
              MODEL=BTFBS, K_Fold=args.fold, LOSS='CrossEntropy')

