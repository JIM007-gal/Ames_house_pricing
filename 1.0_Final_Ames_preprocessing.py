#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:42:01 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Hult University/Academics/Dual Degree/Module B2/Machine Learning/Class 1/2nd_ames

Purpose: To analyze the Ames Housing Dataset in order to publish an 
         intelligence report on Github.

Index: 
    * Data preprocessing: flagging and imputing nas, visual EDA, 
    flagging outliers, and transform categorical variables into dummies.
"""

# Importing packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Loading dataset
housing = pd.read_excel('2.0_Ames Housing Dataset.xls')

housing.head()
housing.shape
housing.info()
housing_describe = housing.describe()



###############################################################################
## 1. Data Preprocessing
###############################################################################

#######################################
# 1.1. Flagging and filling NAs
#######################################

# Flagging
for col in housing:
    if housing[col].isnull().any():
        housing['m_'+col] = housing[col].isnull().astype(int)
 
       
# Plotting NAs columns distributions
plt.subplot(2, 2, 1)
plt.hist(housing['Mas Vnr Area'].dropna())
plt.subplot(2, 2, 2)
plt.hist(housing['Total Bsmt SF'].dropna())
plt.subplot(2, 2, 3)
plt.hist(housing['Garage Cars'].dropna())
plt.subplot(2, 2, 4)
plt.hist(housing['Garage Area'].dropna())
plt.show()


# Filling NAs
housing['Mas Vnr Area'] = housing['Mas Vnr Area'].fillna(0)

fill = housing['Total Bsmt SF'].median()
housing['Total Bsmt SF'] = housing['Total Bsmt SF'].fillna(fill)

fill = housing['Garage Cars'].median()
housing['Garage Cars'] = housing['Garage Cars'].fillna(fill)

fill = housing['Garage Area'].median()
housing['Garage Area'] = housing['Garage Area'].fillna(fill)



#######################################
# 1.2. Visual EDA
#######################################

# List with numerical variables
num_cols = [col for col in housing if np.logical_or(housing[col].dtype == 'int64', housing[col].dtype == 'float64')]
for i in range(4):
    del(num_cols[-1])
del(num_cols[0])


# List with categorical variables
cat_cols = [col for col in housing if housing[col].dtype == 'object']


# Creating folder in current directory to store graphs
cwd = os.getcwd()
os.makedirs(cwd+'/0.0_EDA_Graphs')
graph_path = cwd+'/0.0_EDA_Graphs/'


# 1.2.1. Histograms numerical variables
f, axes = plt.subplots(5, 4, figsize = (24, 20))
for i, e in enumerate(num_cols):
    sns.distplot(housing[e],
                 bins = 'fd' if np.logical_and(e != 'Kitchen AbvGr', e != 'Pool Area') else 15,
                 kde = True,
                 rug = False,
                 ax = axes[i // 4][i % 4])
    plt.xlabel(e)
    
plt.savefig(graph_path + '0_Histograms.png')


# 1.2.2. Scatter plots
f, axes = plt.subplots(5, 4, figsize = (24, 20))
for i, e in enumerate(num_cols):
    sns.scatterplot(x = housing[e], 
                    y = housing['SalePrice'],
                    ax = axes[i // 4][i % 4])
    
plt.savefig(graph_path + '1_Scatter_plots.png')


# 1.2.3. Boxplots
f, axes = plt.subplots(2, 2, figsize = (15, 15))
for i,e in enumerate(cat_cols):
    sns.boxplot(x = housing['SalePrice'],
                y = housing[e],
                ax = axes[i // 2][i % 2])
    sns.set_style('whitegrid')

for c in axes:
    for ax in c:
        ax.axvline(housing['SalePrice'].mean())
        ax.yaxis.grid(True)
        
plt.savefig(graph_path + '2_Boxplots.png')



#######################################
# 1.3. Outlier flagging
#######################################

# Creating a dictionary to store all the outliers
out_list = [{'var': 'Lot Area','lo': 5000,'hi': 23000},
            {'var': 'Overall Qual', 'lo': 3, 'hi': 9},
            {'var': 'Overall Cond', 'lo': 4, 'hi': 9},
            {'var': 'Mas Vnr Area', 'lo': 0, 'hi': 400},
            {'var': 'Total Bsmt SF', 'lo': 0, 'hi': 1800},
            {'var': '1st Flr SF', 'lo': 350, 'hi': 1750},
            {'var': '2nd Flr SF', 'lo': 0, 'hi': 1300},
            {'var': 'Gr Liv Area', 'lo': 600, 'hi': 2500},
            {'var': 'Full Bath', 'lo': 0, 'hi': 3},
            {'var': 'Half Bath', 'lo': -1, 'hi': 2},
            {'var': 'Kitchen AbvGr', 'lo': 0, 'hi': 2},
            {'var': 'TotRms AbvGrd', 'lo': 3, 'hi': 9},
            {'var': 'Fireplaces', 'lo': -1, 'hi': 3},
            {'var': 'Garage Cars', 'lo': 0, 'hi': 4},
            {'var': 'Garage Area', 'lo': 0, 'hi': 900},
            {'var': 'Porch Area', 'lo': 0, 'hi': 450},
            {'var': 'Pool Area', 'lo': -1, 'hi': 100}
            ]


# Creating flag columns
num_cols_out = num_cols.copy()
del(num_cols_out[-1]) 

for i, e in enumerate(num_cols_out):
    housing['out_'+e] = housing[e].apply(lambda x: 1 if x >= out_list[i]['hi'] else 0)
    
# (2 if x >= out_list[i]['hi'] else 1))

#######################################
# 1.4. Converting categorical into dummies
#######################################
    
# One-Hot Encoding qualitative variables
street_dummies = pd.get_dummies(list(housing['Street']), drop_first = True)
lot_config_dummies = pd.get_dummies(list(housing['Lot Config']), drop_first = True)
neighborhood_dummies = pd.get_dummies(list(housing['Neighborhood']), drop_first = True)


# Concatenating in the housing dataset
housing = pd.concat([housing, street_dummies, lot_config_dummies, neighborhood_dummies],
                    axis = 1)

housing = housing.drop(['Street', 'Lot Config', 'Neighborhood'],
                       axis = 1)


# Saving the preprocessed dataset
housing.to_excel('2.1_Housing_preprocessed.xlsx')