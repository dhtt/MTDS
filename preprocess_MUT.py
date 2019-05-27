#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:31:16 2019

@author: dohoangthutrang
"""
import pandas as pd
import sys
import time
start_time = time.time()


def preprocess_mutation_data(mutationdataset, cosmic_id, refgeneslist):
    ref_genes_list = [line.rstrip('\n') for line in open(refgeneslist)] #yield list of ref genes
    c_id_list = map(int, [line.rstrip('\n') for line in open(cosmic_id)])
    with open(mutationdataset) as f:
        mutation_dataset = (pd.read_table(f, sep='\t', index_col=0, lineterminator='\n')
                        .reindex(index = c_id_list, columns = ref_genes_list) #filter rows and columns using cosmic_id and ref genes
                        .dropna() #??? dropna or not
                        )
        mutation_dataset.to_csv('mutation_dataset.csv')
    return(mutation_dataset)

def main(agrv):
    if len(agrv) != 3:
        print('Please check if the mutation dataset, cosmic id and reference gene list have been given.')
    else:
        preprocess_mutation_data(sys.argv[1],sys.argv[2],sys.argv[3])
        print('Success! Check mutation_dataset.csv file.')
        print("--- %s seconds ---" % (time.time() - start_time))
        sys.exit(2)
        
if __name__ == '__main__':
    main(sys.argv[1:])

#MU = preprocess_mutation_data('GDSC_Mutation_2018_matrix.txt', 'cosmic_id.txt', 'intogen_driver_genes_21_03_2019.txt')