#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:42:21 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Hult University/Academics/Dual Degree/Module B2/Machine Learning/Class 1/2nd_ames

Purpose: To analyze the Ames Housing Dataset in order to publish an 
         intelligence report on Github.
         
Index: * Feature Engineering and Linear Regression
       * K Nearest Neighbors Regressor
       * Regression Tree
       * Model Selection
"""

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
import os


# Load preprocessed dataset
housing = pd.read_excel('2.1_Housing_preprocessed.xlsx')


# Redefine cwd
cwd = os.getcwd()



###############################################################################
## 1. Feature Engineering and Linear Regression
###############################################################################

## Full model with statsmodels
full_model = smf.ols(formula = """SalePrice ~ housing['Lot Area'] +
                                              housing['Overall Qual'] +
                                              housing['Overall Cond'] +
                                              housing['Mas Vnr Area'] +
                                              housing['Total Bsmt SF'] +
                                              housing['1st Flr SF'] +
                                              housing['2nd Flr SF'] +
                                              housing['Gr Liv Area'] +
                                              housing['Full Bath'] +
                                              housing['Half Bath'] +
                                              housing['Kitchen AbvGr'] +
                                              housing['TotRms AbvGrd'] +
                                              housing['Fireplaces'] +
                                              housing['Garage Cars'] +
                                              housing['Garage Area'] +
                                              housing['Porch Area'] +
                                              housing['Pool Area'] +
                                              housing['m_Mas Vnr Area'] +
                                              housing['m_Total Bsmt SF'] +
                                              housing['m_Garage Cars'] +
                                              housing['m_Garage Area'] +
                                              housing['out_Lot Area'] +
                                              housing['out_Overall Qual'] +
                                              housing['out_Overall Cond'] +
                                              housing['out_Half Bath'] +
                                              housing['out_Kitchen AbvGr'] +
                                              housing['out_TotRms AbvGrd'] +
                                              housing['out_Fireplaces'] +
                                              housing['out_Garage Cars'] +
                                              housing['out_Garage Area'] +
                                              housing['out_Porch Area'] +
                                              housing['out_Pool Area'] +
                                              housing['Pave'] +
                                              housing['CulDSac'] +
                                              housing['FR2'] +
                                              housing['FR3'] +
                                              housing['Inside'] +
                                              housing['Blueste'] +
                                              housing['BrDale'] +
                                              housing['BrkSide'] +
                                              housing['ClearCr'] +
                                              housing['CollgCr'] +
                                              housing['Crawfor'] +
                                              housing['Edwards'] +
                                              housing['Gilbert'] +
                                              housing['Greens'] +
                                              housing['GrnHill'] +
                                              housing['IDOTRR'] +
                                              housing['Landmrk'] +
                                              housing['MeadowV'] +
                                              housing['Mitchel'] +
                                              housing['NAmes'] +
                                              housing['NPkVill'] +
                                              housing['NWAmes'] +
                                              housing['Somerst'] +
                                              housing['StoneBr'] +
                                              housing['Timber'] +
                                              housing['Veenker'] - 1
                                              """,
                    data = housing)

results = full_model.fit()
print(results.summary())


# Saving results in a csv file
os.makedirs(cwd+'/0.1_First_Regression')
first_reg_path = cwd+'/0.1_First_Regression/'

beginning_text_1 = """documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
end_text_1 = '\end{document}'

f = open(first_reg_path + '0_1st_Regression.csv', 'w')
f.write(beginning_text_1)
f.write(results.summary().as_csv())
f.write(end_text_1)
f.close()


# Residual plot to identify anomalies in the residuals
plt.scatter(results.fittedvalues, results.resid)
plt.savefig(first_reg_path+'1_Residuals.png')
plt.show()


## Significant model 1
significant_1_model = smf.ols(formula = """SalePrice ~    housing['Lot Area'] +
                                                          housing['Overall Qual'] +
                                                          housing['Overall Cond'] +
                                                          housing['Mas Vnr Area'] +
                                                          housing['Total Bsmt SF'] +
                                                          housing['1st Flr SF'] +
                                                          housing['2nd Flr SF'] +
                                                          housing['Full Bath'] +
                                                          housing['Half Bath'] +
                                                          housing['TotRms AbvGrd'] +
                                                          housing['Fireplaces'] +
                                                          housing['Garage Cars'] +
                                                          housing['Garage Area'] +
                                                          housing['Porch Area'] +
                                                          housing['out_Lot Area'] +
                                                          housing['out_Overall Qual'] +
                                                          housing['out_Overall Cond'] +
                                                          housing['out_Half Bath'] +
                                                          housing['out_Kitchen AbvGr'] +
                                                          housing['out_TotRms AbvGrd'] +
                                                          housing['out_Fireplaces'] +
                                                          housing['out_Garage Cars'] +
                                                          housing['out_Garage Area'] +
                                                          housing['out_Porch Area'] +
                                                          housing['BrDale'] +
                                                          housing['Crawfor'] +
                                                          housing['Edwards'] +
                                                          housing['Gilbert'] +
                                                          housing['GrnHill'] +
                                                          housing['IDOTRR'] +
                                                          housing['NAmes'] +
                                                          housing['NPkVill'] +
                                                          housing['NWAmes'] +
                                                          housing['StoneBr'] - 1
                                                          """,
                                data = housing)

results_s1 = significant_1_model.fit()
print(results_s1.summary())


# Creating a file to save the results
os.makedirs(cwd+'/0.2_Second_Regression')
sec_reg_path = cwd+'/0.2_Second_Regression/'

beginning_text_2 = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
end_text_2 = '\end{document}'

f = open(sec_reg_path+'0_2nd_Regression.csv','w')
f.write(beginning_text_2)
f.write(results_s1.summary().as_csv())
f.write(end_text_2)
f.close()


# Collinearity analysis
improv = ['Lot Area',
          'Overall Qual',
          'Overall Cond',
          'Mas Vnr Area',
          'Total Bsmt SF',
          '1st Flr SF',
          '2nd Flr SF',
          'Full Bath',
          'Half Bath',
          'TotRms AbvGrd',
          'Fireplaces',
          'Garage Cars',
          'Garage Area',
          'Porch Area',
          'out_Lot Area',
          'out_Overall Qual',
          'out_Overall Cond',
          'out_Half Bath',
          'out_Kitchen AbvGr',
          'out_TotRms AbvGrd',
          'out_Fireplaces',
          'out_Garage Cars',
          'out_Garage Area',
          'out_Porch Area',
          'BrDale',
          'Crawfor',
          'Edwards',
          'Gilbert',
          'GrnHill',
          'IDOTRR',
          'NAmes',
          'NPkVill',
          'NWAmes',
          'StoneBr'
          ]


# Variance Inflation Factor to identify collinearity
ck = np.column_stack([np.array(housing.loc[:,col]) for col in improv])
VIF = pd.DataFrame([variance_inflation_factor(ck,i) for i in range(ck.shape[1])])
VIF.index = [col for col in improv]
VIF.to_excel(sec_reg_path+'1_VIF.xlsx')


# Correlation heatmap
df_corr = housing.loc[:,improv].corr().round(2)
df_corr.to_excel(sec_reg_path+'2_Correlation matrix.xlsx')

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df_corr.iloc[1:19,1:19],
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig(sec_reg_path+'3_Correlation heatmap.png')
plt.show()


# Residual plot to identify residual anomalies
plt.scatter(results_s1.fittedvalues,results.resid)

plt.savefig(sec_reg_path+'4_Residual plot.png')
plt.show()


# Influence plot to analyze outliers and high leverage points
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(results_s1, alpha  = 0.05, ax = ax, criterion = "cooks")

plt.savefig(sec_reg_path+'5_Influence plot.png')
plt.show()


## Linear Regression with sklearn
# Separating dependent and independent variables
housing_data = housing[improv]
housing_target = housing.loc[:,'SalePrice']


# Splitting the set in train and validation
X_train, X_valid, y_train, y_valid = train_test_split(housing_data,
                                                    housing_target,
                                                    test_size = 0.25,
                                                    random_state = 992)

lr = LinearRegression()
lr_fit = lr.fit(X_train, y_train)
y_ols_validscore = lr_fit.score(X_valid, y_valid)

print('Training score', lr_fit.score(X_train, y_train).round(4))
print('Validation score', y_ols_validscore.round(4))
print('Cross validation score', cross_val_score(lr, housing_data, housing_target, cv = 3).mean().round(4))


## Significant model 2
housing = housing.drop([1182, 2181, 2180, 1498, 2256, 2892])
housing['crawlot'] = housing['Lot Area'] * housing['Crawfor']
housing['nameslot'] = housing['Lot Area'] * housing['NAmes']
housing['stonelot'] = housing['Lot Area'] * housing['StoneBr']
significant_2_model = smf.ols(formula = """SalePrice ~    housing['Lot Area'] +
                                                          housing['Overall Qual'] +
                                                          housing['Overall Cond'] +
                                                          housing['Mas Vnr Area'] +
                                                          housing['Total Bsmt SF'] +
                                                          housing['1st Flr SF'] +
                                                          housing['2nd Flr SF'] +
                                                          housing['Full Bath'] +
                                                          housing['Half Bath'] +
                                                          housing['Fireplaces'] +
                                                          housing['Garage Cars'] +
                                                          housing['Garage Area'] +
                                                          housing['Porch Area'] +
                                                          housing['out_Lot Area'] +
                                                          housing['out_Overall Qual'] +
                                                          housing['out_Half Bath'] +
                                                          housing['out_Kitchen AbvGr'] +
                                                          housing['out_TotRms AbvGrd'] +
                                                          housing['out_Garage Area'] +
                                                          housing['BrDale'] +
                                                          housing['crawlot'] +
                                                          housing['Edwards'] +
                                                          housing['IDOTRR'] +
                                                          housing['nameslot'] +
                                                          housing['NPkVill'] +
                                                          housing['NWAmes'] +
                                                          housing['stonelot'] - 1
                                                          """,
                                data = housing)

results_s2 = significant_2_model.fit()
print(results_s2.summary())


# Creating a file to save the results
#os.makedirs(cwd+'/0.3_Third_Regression')
third_reg_path = cwd+'/0.3_Third_Regression/'

beginning_text_3 = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
end_text_3 = '\end{document}'

f = open(third_reg_path+'0_3nd_Regression.csv','w')
f.write(beginning_text_3)
f.write(results_s2.summary().as_csv())
f.write(end_text_3)
f.close()


# Collinearity analysis
improv1 = ['Lot Area',
          'Overall Qual',
          'Overall Cond',
          'Mas Vnr Area',
          'Total Bsmt SF',
          '1st Flr SF',
          '2nd Flr SF',
          'Full Bath',
          'Half Bath',
          'Fireplaces',
          'Garage Cars',
          'Garage Area',
          'Porch Area',
          'out_Lot Area',
          'out_Overall Qual',
          'out_Half Bath',
          'out_Kitchen AbvGr',
          'out_TotRms AbvGrd',
          'out_Garage Area',
          'BrDale',
          'crawlot',
          'Edwards',
          'IDOTRR',
          'nameslot',
          'NPkVill',
          'NWAmes',
          'stonelot'
          ]


# Variance Inflation Factor to identify collinearity
ck1 = np.column_stack([np.array(housing.loc[:,col]) for col in improv1])
VIF1 = pd.DataFrame([variance_inflation_factor(ck1,i) for i in range(ck1.shape[1])])
VIF1.index = [col for col in improv1]
VIF1.to_excel(third_reg_path+'1_VIF.xlsx')

# Correlation heatmap
df_corr1 = housing.loc[:,improv1].corr().round(2)
df_corr1.to_excel(third_reg_path+'2_Correlation matrix.xlsx')

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(df_corr1.iloc[1:19,1:19],
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5)


plt.savefig(third_reg_path+'3_Correlation heatmap.png')
plt.show()

# Residual plot to identify residual anomalies
plt.scatter(results_s2.fittedvalues,results_s2.resid)

plt.savefig(third_reg_path+'4_Residual plot.png')
plt.show()

# Influence plot to analyze outliers and high leverage points
fig, ax = plt.subplots(figsize=(12,8))
fig = sm.graphics.influence_plot(results_s2, alpha  = 0.05, ax = ax, criterion = "cooks")

plt.savefig(third_reg_path+'5_Influence plot.png')
plt.show()

## Linear Regression with sklearn
# Separating dependent and independent variables
housing_data = housing[improv1]
housing_target = housing.loc[:,'SalePrice']

# Splitting the set in train and validation
X_train, X_valid, y_train, y_valid = train_test_split(housing_data,
                                                    housing_target,
                                                    test_size = 0.25,
                                                    random_state = 992)


# Fitting and scoring the Linear Regression model
lr1 = LinearRegression()
lr_fit1 = lr1.fit(X_train, y_train)
y_ols_validscore1 = lr_fit1.score(X_valid, y_valid)


# Saving scores in a dictionary as part of a comparing process in feature selection
score_names = ['Training score', 'Validation score', 'Cross validation score']
scores = [[lr_fit1.score(X_train, y_train).round(4)],
          [y_ols_validscore1.round(4)],
          [cross_val_score(lr1, housing_data, housing_target, cv = 3).mean().round(4)]]

zipscores = zip(score_names, scores)
scores_dict = dict(zipscores)


# How well does our model perform fitted with different datasets?
y_crossscore_ols_optimal = cross_val_score(lr1, housing_data, housing_target, cv = 3).mean().round(4)



###############################################################################
## 2. k Nearest Neighbors
###############################################################################

#########################
# 2.1. Build a 'learning base' with KNN on the full model
#########################

# Train/Test split
housing_data = housing.drop(['SalePrice'], axis = 1)
housing_target = housing['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(housing_data, 
                                                    housing_target,
                                                    test_size = 0.25,
                                                    random_state = 992)

## Scaling our Train/Test split

feature_scaler = StandardScaler()
feature_scaler.fit(X_train)
X_train = feature_scaler.transform(X_train)
X_test = feature_scaler.transform(X_test)


# In case we don't have the test data, we apply GridSearchCV to select optimal k
k_space = pd.np.arange(1,21)
param_grid = {'n_neighbors': k_space}

knn_grid = KNeighborsRegressor()
knn_grid_cv = GridSearchCV(knn_grid, param_grid, cv = 3)
knn_grid_cv.fit(X_train, y_train)

print('Best parameters:', knn_grid_cv.best_params_)
print('Cross Validation score:', knn_grid_cv.best_score_.round(4))


# Maximum score at k = 4
knn_base = KNeighborsRegressor(n_neighbors = 4)
knn_base.fit(X_train, y_train)
y_score_optimal_knn_base = knn_base.score(X_test, y_test).round(4)

print('Training score:', knn_base.score(X_train, y_train).round(4))
print('Testing/ Validation score:', y_score_optimal_knn_base)



#########################
# 2.2. Optimal KNN model with significant variables
#########################

# Train/Test split
housing_data = housing.loc[:, improv1]
housing_target = housing['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(housing_data,
                                                    housing_target,
                                                    test_size = 0.25,
                                                    random_state = 992)

# Scaling our split
feature_scaler = StandardScaler()
feature_scaler.fit(X_train)
X_train = feature_scaler.transform(X_train)
X_test = feature_scaler.transform(X_test)


# Selecting optimal k
k_space = pd.np.arange(1,21)
param_grid = {'n_neighbors': k_space}

knn_grid = KNeighborsRegressor()
knn_grid_cv = GridSearchCV(knn_grid, param_grid, cv = 3)
knn_grid_cv.fit(X_train, y_train)

print('Best parameters:', knn_grid_cv.best_params_)
print('Cross Validation score:', knn_grid_cv.best_score_.round(4))


# Maximum score at k = 4
knn = KNeighborsRegressor(n_neighbors = 4)
knn_fit = knn.fit(X_train, y_train)
y_score_optimal_knn = knn_fit.score(X_test, y_test).round(4)

print('Training score:', knn_fit.score(X_train, y_train).round(4))
print('Testing/ Validation score:', y_score_optimal_knn)
y_crossscore_knn_optimal = cross_val_score(knn, housing_data, housing_target, cv = 3).mean().round(4)



###############################################################################
## 3. Decision Tree with scikit-learn
###############################################################################

# os.makedirs(cwd + '/0.5_Regression_tree')
tree_path = cwd + '/0.5_Regression_tree/'


# Train/Test split
housing_data = housing.loc[:, improv1]
housing_target = housing['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(housing_data,
                                                    housing_target,
                                                    test_size = 0.25,
                                                    random_state = 992)


# Building tree model
depth_space = pd.np.arange(1,15)
leaf_space = pd.np.arange(25,50)
param_grid = {'max_depth': depth_space,
              'min_samples_leaf': leaf_space}

tree_grid = DecisionTreeRegressor(random_state = 992)
tree_grid_cv = GridSearchCV(tree_grid, param_grid, cv = 3)
tree_grid_cv.fit(X_train, y_train)

print('Best parameters:', tree_grid_cv.best_params_)
print('Cross Validation score:', tree_grid_cv.best_score_.round(4))


# Maximum score at Max depth = 10, Min samples leaf = 28
tree = DecisionTreeRegressor(criterion = 'mse', 
                             max_depth = 10,
                             min_samples_leaf = 28,
                             random_state = 992)

tree_fit = tree.fit(X_train, y_train)


y_score_tree_optimal = tree.score(X_test, y_test).round(4)
print('Training score:', tree.score(X_train, y_train).round(4))
print('Testing/ Validation score:', y_score_tree_optimal)
y_crossscore_tree_optimal = cross_val_score(tree, housing_data, housing_target, cv = 3).mean().round(4)


# Plot tree
dot_data = StringIO()

export_graphviz(decision_tree = tree,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = housing_data.columns)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)

graph.write_png(tree_path+'Ames_Decision_Tree.png')


# Plot feature importance
def plot_feature_importance(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize = (12,9))
    n_features = train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align = 'center')
    plt.yticks(np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    if export == True:
        plt.savefig(tree_path+"Tree_feature_importance.png")
        
plot_feature_importance(tree, train = X_train, export = True)



###############################################################################
## 4. Model selection and predictions
###############################################################################


# Create a new file in directory
# os.makedirs(cwd + '/0.6_Results_interpretability')
interp_path = cwd + '/0.6_Results_interpretability/'
    

# Print and save model results
print(f"""
Optimal model KNN score: {y_crossscore_knn_optimal.round(3)}
Optimal model OLS score: {y_crossscore_ols_optimal.round(3)}
Optimal model Tree score: {y_crossscore_tree_optimal.round(3)}
""")

cross_val_scores = {'Cross validation scores': [y_crossscore_knn_optimal.round(3),
                                                y_crossscore_ols_optimal.round(3),
                                                y_crossscore_tree_optimal.round(3)]}

cross_index = ['KNN', 'OLS', 'Tree']

model_sel_scores = pd.DataFrame(cross_val_scores)
model_sel_scores.index = cross_index

model_sel_scores.to_excel(interp_path + '0_Scores.xlsx')

'''
Taking into consideration the previous results, OLS is the method that
performs the best when comparing it against KNN and Regression Tree. This
fact is due to the nature of the problem. To classify properties into
categories is not the most suitable approach when predicting, as trees do.
However, many houses include bedrooms, kitchens, bathrooms,... and their number
constraints the final price of the house when multiplied by a coefficient. The
latter is modelled the best by Linear Regression (OLS).
'''

## Save predictions
# Reloading housing dataset to account for all observations
housing = pd.read_excel('2.1_Housing_preprocessed.xlsx')
housing['crawlot'] = housing['Lot Area'] * housing['Crawfor']
housing['nameslot'] = housing['Lot Area'] * housing['NAmes']
housing['stonelot'] = housing['Lot Area'] * housing['StoneBr']

housing_data = housing.drop(['SalePrice'], axis = 1).loc[:, improv1]

# Build dataframes to later concatenate them and compare against actual values
lr_fit1_pred = pd.DataFrame({'OLS': lr_fit1.predict(housing_data).astype(int)})
tree_fit_pred = pd.DataFrame({'Tree': tree_fit.predict(housing_data).astype(int)})

feature_scaler = StandardScaler()
feature_scaler.fit(housing_data)
housing_data = feature_scaler.transform(housing_data)
knn_fit_pred = pd.DataFrame({'KNN': knn_fit.predict(housing_data).astype(int)})

predictions = pd.concat([housing['SalePrice'], lr_fit1_pred, knn_fit_pred, tree_fit_pred],
                        axis = 1)

predictions.to_excel(interp_path + '1_Predictions.xlsx')


