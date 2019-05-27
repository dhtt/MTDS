#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:08:07 2019

@author: dohoangthutrang
"""
import pandas as pd
import collections
import re
import time
import sys
start_time = time.time()


def merge_duplicate_drug(df):
    drug_list = list(map(lambda s: str(s).split('_')[0], df.columns)) #yield list of drug names from splitting col names by tab
    df_sub = df.copy()
    df_sub.columns = drug_list
    drug_count = collections.Counter(df_sub.columns) #return frequency of drug in list
    dup_drug = [drug for drug,count in drug_count.items() if count==2] #return drugs that are screened twice
    for i in dup_drug:
        dup_col = df_sub[i]
        df_sub = df_sub.drop(dup_col, axis = 1) #drop all columns of dup drug
        df_sub[i] = dup_col.mean(axis = 1) #and add new col containing the mean of dropped cols for that drug 
    return(df_sub)
        
def preprocess_drug_data(drugdataset):
    with open(drugdataset) as f:
        drug_dataset = pd.read_table(f, sep= '\t', index_col=0, lineterminator='\n')
        drug_dataset = merge_duplicate_drug(drug_dataset)
        drug_dataset = drug_dataset.dropna(axis = 0) #??? dropna first or merge dup drugs first/ dropna or not
        drug_dataset.to_csv('drug_dataset.csv')
        cosmic_id = list(drug_dataset.index)
        with open("cosmic_id.txt", "w") as output:
            for i in cosmic_id:
                output.write("%s\n" % i) #Write cos_id into a txt file
    return(drug_dataset)

def main(agrv):
    if len(agrv) != 1:
        print('Please check of the drug dataset has been given.')
    else:
        preprocess_drug_data(sys.argv[1])
        print('Success! Check drug_dataset.csv file.')
        print("--- %s seconds ---" % (time.time() - start_time))
        sys.exit(2)
        
if __name__ == '__main__':
    main(sys.argv[1:])