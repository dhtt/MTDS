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
import math
import numpy as np
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
        df  = pd.DataFrame(drug_dataset)   #change dataset into Pandas Dataframs (Easier to manipulate) 
        while df.isnull().values.any():     # while the data set has even a single NA value
            print(df.shape)                 # print the shape of the data frame
            df = removeDominantNA(df)       # Removes the column or row with the largest amount of NAs based on a penalizing value
        newDrug = df
        print('is NA', df.isnull().values.any())
        newDrug.to_csv('drug_dataset.csv')  # writes output to CSV
        cosmic_id = list(newDrug.index)     # Id's of cell lines remaining in drug dataset
        with open("cosmic_id.txt", "w") as output:
            for i in cosmic_id:
                output.write("%s\n" % i) #Write cos_id into a txt file
    return(newDrug)
def removeDominantNA(df):
    numberOfNanColumns = np.zeros((df.shape[1],), dtype=int) 
    numberOfNanRows = np.zeros((df.shape[0],), dtype=int) 
    print('starting')
    for i in range(len(df)):    # Finds the number of NAs w.r.t Rows and columns
        for j in range(len(df.iloc[i])):
            if math.isnan(df.iloc[i][j]):
                   numberOfNanColumns[j] = numberOfNanColumns[j]+1
                   numberOfNanRows[i] = numberOfNanRows[i]+1
    maxValueColumn=0
    maxIndexColumn=0
    for i in range(len(numberOfNanColumns)):    # Finds Column with maximum number of NAs
        if maxValueColumn < numberOfNanColumns[i]:
            maxValueColumn = numberOfNanColumns[i]
            maxIndexColumn = i
            
    print('max Nan Column value,index',maxValueColumn,maxIndexColumn)
    maxValueRow=0
    maxIndexRow=0
    for i in range(len(numberOfNanRows)):   # Finds row with maximum number of NAs
        if maxValueRow < numberOfNanRows[i]:
            maxValueRow = numberOfNanRows[i]
            maxIndexRow = i
            
    print('max Nan Row value,index',maxValueRow,maxIndexRow)
    print('')
    print('Ratio row column',maxValueRow/df.shape[1],' ',maxValueColumn/df.shape[0])
    if maxValueRow/df.shape[1] > ( 2.5 * maxValueColumn/df.shape[0]):   #Decides whether to remove the column or the row. Since we want around a 100 drugs, the weight term was chosen by trial and error
        df1 = remove_Dominant_NAN_row(df,maxIndexRow)                   # Decreasing the weight term increases the number of columns but decreases the number of rows. 
    else:
        df1 = remove_Dominant_NAN_column(df,maxIndexColumn)
    return(df1)

def remove_Dominant_NAN_row(df,i):
    print('deleting row ', i)
    df = df.drop(df.index[i])
    return(df)

def remove_Dominant_NAN_column(df,i):
    print('deleting Column ', i)
    df = df.drop(df.columns[i], axis=1)
    return(df)

drug_data2_5_New = preprocess_drug_data('../../GDSC_Drug_2018_matrix.txt')

#def main(agrv):
#    if len(agrv) != 1:
#        print('Please check of the drug dataset has been given.')
#    else:
#        preprocess_drug_data(sys.argv[1])
#        print('Success! Check drug_dataset.csv file.')
#        print("--- %s seconds ---" % (time.time() - start_time))
#        sys.exit(2)
#        
#if __name__ == '__main__':
#    main(sys.argv[1:])