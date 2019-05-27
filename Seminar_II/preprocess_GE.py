#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:36:06 2019

@author: dohoangthutrang
"""
import pandas as pd
import sys
import time
import mygene 
start_time = time.time()

def get_ensembl_id(refgeneslist):
    mg = mygene.MyGeneInfo()
    ref_genes_list = [line.rstrip('\n') for line in open(refgeneslist)]
    HGNC_to_ENSG = mg.querymany(ref_genes_list, scopes="symbol", fields=["ensembl.gene"], species="human", as_dataframe=True, returnall = False)['ensembl.gene']
    HGNC_to_ENSG = HGNC_to_ENSG.dropna()
    HGNC_to_ENSG = HGNC_to_ENSG.to_dict() #return dict(key = HGNC, val = ENSG)
    return(HGNC_to_ENSG)

#ENSG = get_ensembl_id('intogen_driver_genes_21_03_2019.txt')

def preprocess_gene_expression_data(geneexpressiondataset, cosmic_id,  refgeneslist):
    ENSG_list = get_ensembl_id(refgeneslist).values()
    HGNC_list = list(get_ensembl_id(refgeneslist).keys())
    c_id_list = [line.rstrip('\n') for line in open(cosmic_id)]
    with open(geneexpressiondataset) as f:
        ge_dataset = (pd.read_table(f, sep= '\t', index_col=0, lineterminator='\n')
                    .reindex(columns = c_id_list, index = ENSG_list)
                    )
        ge_dataset.index = HGNC_list
        ge_dataset = ge_dataset.dropna()
        ge_dataset = ge_dataset.transpose()
        ge_dataset.to_csv('ge_dataset.csv')
    return(ge_dataset)

def main(agrv):
    if len(agrv) != 3:
        print('Please check if the CNV dataset, cosmic id and reference gene list have been given.')
    else:
        preprocess_gene_expression_data(sys.argv[1],sys.argv[2],sys.argv[3])
        print('Success! Check ge_dataset.csv file.')
        print("--- %s seconds ---" % (time.time() - start_time))
        sys.exit(2)
        
if __name__ == '__main__':
    main(sys.argv[1:])    
    
#GE = read_gene_expression_data('GDSC_GeneExpression_normalized_2018_matrix.txt','cosmic_id.txt', 'intogen_driver_genes_21_03_2019.txt')
