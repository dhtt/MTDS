# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            print('Column name : ', col)
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
 
    return list(duplicateColumnNames)


gene_expression = pd.read_table("GDSC_GeneExpression_normalized_2018_matrix.txt")
gene_expression_transposed = gene_expression.transpose()
copy = pd.read_table("GDSC_CNV_2018_matrix.txt")
drug = pd.read_table("GDSC_Drug_2018_matrix.txt")
copy_NAN_removed = copy[~np.isnan(copy).any(axis=1)]
drug_NAN_removed = drug[~np.isnan(drug).any(axis=1)]
gene_NAN_removed = gene_expression_transposed[pd.notnull(gene_expression_transposed).any(axis=1)]
Duplicated_Drug_Columns = drug_NAN_removed.loc[:,~drug_NAN_removed.columns.duplicated()]

"""print('Duplicated Columns:' + Duplicated_Drug_Columns)

# Get list of duplicate columns
duplicateColumnNames = getDuplicateColumns(drug_NAN_removed)
 
print('Duplicate Columns are as follows')
for col in duplicateColumnNames:
    print('Column name : ', col)



"""
