#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:08:26 2019

@author: dohoangthutrang
"""
import pandas as pd
import numpy as np

def check_any_null(dataset_list):
    for dataset in dataset_list:
        print(dataset.isnull().values.any())
        print(np.any(np.isnan(dataset)))

def get_unmutual_celllines(dataset_list):
    celllines_list = []
    for dataset1 in dataset_list:
        a = dataset1.index.values.tolist()
        for dataset2 in dataset_list:
            b = dataset2.index.values.tolist()
            diff = list(set(a) - set(b))
            for i in diff:
                celllines_list.append(i)
    return celllines_list

def get_processed_data(dataset, cosmic_id, filename):
    dataset=dataset.reindex(index = cosmic_id)
    dataset.to_csv(filename)
GE = pd.read_csv('ge_dataset.csv', delimiter=',', index_col = 0) 
CNV = pd.read_csv('CNV_dataset.csv', delimiter=',', index_col = 0)
MUT = pd.read_csv('mutation_dataset.csv', delimiter=',', index_col = 0)
DRUG = pd.read_csv('drug_dataset.csv', delimiter=',', index_col = 0)

dataset_list = [GE,CNV,MUT,DRUG]
cosmic_id = get_unmutual_celllines(dataset_list)
for dataset in dataset_list:
    index = list(set(dataset.index.values.tolist()) - set(cosmic_id))
    print(len(index)) #Check if len of datasets are the same after removal of cell

CNV = CNV.reindex(index = index)
#CNV.to_csv('processed_CNV1.csv')
DRUG = DRUG.reindex(index = index)
#DRUG.to_csv('processed_DR1.csv')
GE = GE.reindex(index = index)
#GE.to_csv('processed_GE1.csv')
MUT = MUT.reindex(index = index)
#MUT.to_csv('processed_MUT1.csv')