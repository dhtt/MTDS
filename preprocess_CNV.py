#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:34:38 2019

@author: dohoangthutrang
"""
import pandas as pd
import sys
import re 
import time
start_time = time.time()

def preprocess_cnv_data(cnvdataset, cosmic_id, refgeneslist):
    ref_genes_list = [line.rstrip('\n') for line in open(refgeneslist)] #yield list of ref genes
    ref_genes_list = re.compile(str("^("+'|'.join(ref_genes_list)+")_"))
    c_id_list = map(int, [line.rstrip('\n') for line in open(cosmic_id)])
    with open(cnvdataset) as f:
        cnv_dataset = (pd.read_table(f, sep='\t', index_col=0, lineterminator='\n')
                    .filter(regex = ref_genes_list, axis = 1) #filter columns using ref genes
                    .reindex(index = c_id_list) #filter rows using cosmic_id
                    .dropna() #??? dropna or not
                    )
        cnv_loss = cnv_dataset.filter(regex = 'loss', axis = 1)
        cnv_gain = cnv_dataset.filter(regex = 'gain', axis = 1)
        cnv_loss.to_csv('cnv_loss.csv')
        cnv_gain.to_csv('cnv_gain.csv')
        cnv_dataset.to_csv('cnv_dataset.csv')
    return(cnv_loss, cnv_gain, cnv_dataset)
 
def main(agrv):
    if len(agrv) != 3:
        print('Please check if the CNV dataset, cosmic id and reference gene list have been given.')
    else:
        preprocess_cnv_data(sys.argv[1],sys.argv[2],sys.argv[3])
        print('Success! Check CNV_loss.csv , CNV_gain.csv and CNV_dataset.csv files.')
        print("--- %s seconds ---" % (time.time() - start_time))
        sys.exit(2)
        
if __name__ == '__main__':
    main(sys.argv[1:])    
    
    
#CNV_loss = preprocess_cnv_data('GDSC_CNV_2018_matrix.txt','cosmic_id.txt','tcga_oncogenic_signaling_pathway_cnv_drivers_21_03_2019.txt')[0]
#CNV_gain = preprocess_cnv_data('GDSC_CNV_2018_matrix.txt','cosmic_id.txt','tcga_oncogenic_signaling_pathway_cnv_drivers_21_03_2019.txt')[1]

