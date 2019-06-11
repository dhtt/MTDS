#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:08:26 2019

@author: dohoangthutrang
"""
import pandas as pd

def check_any_null(dataset_list):
    for dataset in dataset_list:
        print(dataset.isnull().values.any())

def get_mutual_celllines(dataset_list):
    celllines_list = []
    for dataset in dataset_list:
        celllines_list.append(dataset.index.values)
    cosmic_id = min(celllines_list, key=len).tolist()
    for l in celllines_list:
        for i in cosmic_id:
            if i not in l:
                cosmic_id.remove(i)
    return cosmic_id

def get_processed_data(dataset, cosmic_id, filename):
    dataset=dataset.reindex(index = cosmic_id)
    dataset.to_csv(filename)
GE = pd.read_csv('ge_dataset.csv', delimiter=',', index_col = 0) 
CNV = pd.read_csv('CNV_dataset.csv', delimiter=',', index_col = 0)
MUT = pd.read_csv('mutation_dataset.csv', delimiter=',', index_col = 0)
DRUG = pd.read_csv('drug_dataset.csv', delimiter=',', index_col = 0)

dataset_list = [GE,CNV,MUT,DRUG]
cosmic_id = get_mutual_celllines(dataset_list)

CNV = CNV.reindex(index = cosmic_id)
CNV.to_csv('processed_CNV.csv')
DRUG = DRUG.reindex(index = cosmic_id)
DRUG.to_csv('processed_DR.csv')
GE = GE.reindex(index = cosmic_id)
GE.to_csv('processed_GE.csv')
MUT = MUT.reindex(index = cosmic_id)
MUT.to_csv('processed_MUT.csv')