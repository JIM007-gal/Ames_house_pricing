#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 17:42:44 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Hult University/Academics/Dual Degree/Module B2/Machine Learning/Class 1/2nd_ames

Purpose: To analyze the Ames Housing Dataset in order to publish an 
         intelligence report on Github.
         
Index: * Random Forest Tuning
       * GBM Tuning
       * RF fitting and score + Feature importance
       * GBM fitting and score + Feature importance       
"""

# Loading libraries
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
import os


# Load preprocessed dataset
housing = pd.read_excel('2.1_Housing_preprocessed.xlsx')


# Create file to save all the results
cwd = os.getcwd()
os.makedirs(cwd + '/0.7_Ensemble_models')
ensemble_path = cwd + '/0.7_Ensemble_models/'



###############################################################################
## 1. Random Forest
###############################################################################

# Splitting the data
housing = housing.drop([1182, 2181, 2180, 1498, 2256, 2892])
housing_data = housing.drop(['SalePrice'], axis = 1)
housing_target = housing['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(housing_data,
                                                    housing_target,
                                                    test_size = 0.25,
                                                    random_state = 992)

# Creating a parameter grid
estimators_space = pd.np.arange(100, 1000, 50)
leaf_space = pd.np.arange(1, 150, 15)
features_space = ['auto', 'sqrt', 'log2']

param_grid = {'n_estimators': estimators_space,
              'min_samples_leaf': leaf_space,
              'max_features': features_space
             }
    

# Choosing parameters with GridSearchCV to tune the Random Forest
rf_grid = RandomForestRegressor(random_state = 992)
rf_grid_cv = GridSearchCV(rf_grid, param_grid, cv = 3)
rf_grid_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print('Tuned RF parameters:', rf_grid_cv.best_params_)
print('Tuned RF score:', rf_grid_cv.best_score_.round(4))


# Generating the optimized RF model
rf_opt = RandomForestRegressor(n_estimators = 700,
                               min_samples_leaf = 1,
                               max_features = 'sqrt')

rf_opt_fit = rf_opt.fit(X_train, y_train)
rf_crossvalscore = cross_val_score(rf_opt, housing_data, housing_target, cv = 3).mean().round(4)


########################
# Feature importance function
########################

def plot_feature_importances(model, train = X_train, export = False, name = ''):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig(ensemble_path + name + '_Feature_Importance_Output.png')
      
        
# Plotting and saving feature importance
plot_feature_importances(rf_opt_fit,
                         train = X_train,
                         export = True,
                         name = '0_RF')


###############################################################################
## 2. Gradient Boosted Machines
###############################################################################

# Splitting the data
housing_data = housing.drop(['SalePrice'], axis = 1)
housing_target = housing['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(housing_data,
                                                    housing_target,
                                                    test_size = 0.25,
                                                    random_state = 992)


# Creating a parameter grid
learning_space = pd.np.arange(0.01, 0.22, 0.05)
estimator_space = pd.np.arange(400, 601, 100)
depth_space = pd.np.arange(3,7)
leaf_space = pd.np.arange(1, 32, 15)

param_grid = {'learning_rate': learning_space,
              'n_estimators': estimator_space,
              'max_depth': depth_space,
              'min_samples_leaf': leaf_space}


# Choosing optimal parameters with GridSearchCV to tune the GBM
gbm_grid = GradientBoostingRegressor(random_state = 992)
gbm_grid_cv = GridSearchCV(gbm_grid, param_grid, cv = 3)
gbm_grid_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print('Tuned GBM parameters:', gbm_grid_cv.best_params_)
print('Tuned GBM score:', gbm_grid_cv.best_score_.round(4))


# Generating the optimized GBM model
gbm_opt = GradientBoostingRegressor(learning_rate = 0.11,
                                    n_estimators = 500,
                                    max_depth = 4,
                                    min_samples_leaf = 1)

gbm_opt_fit = gbm_opt.fit(X_train, y_train)
gbm_crossvalscore = cross_val_score(gbm_opt, housing_data, housing_target, cv = 3).mean().round(4)


# Plotting and saving feature importances
plot_feature_importances(gbm_opt_fit,
                         train = X_train,
                         export = True,
                         name = '1_GBM')



########################
# Models recap
########################

print(f"""
Optimal RF score: {rf_crossvalscore}
Optimal GBM score: {gbm_crossvalscore}
""")

# os.makedirs(cwd + '/0.8_Results_predictive')
predict_path = cwd + '/0.8_Results_predictive/'


final_scores = pd.DataFrame({'Cross validation scores': [rf_crossvalscore,
                                                         gbm_crossvalscore]})
final_scores.index = ['RF', 'GBM']
final_scores.to_excel(predict_path + '0_Scores.xlsx')

# Save predictions
housing = pd.read_excel('2.1_Housing_preprocessed.xlsx')

ensemble_pred = pd.DataFrame({'RF': rf_opt_fit.predict(housing_data).astype(int),
                              'GBM': gbm_opt_fit.predict(housing_data).astype(int)})

predictions = pd.concat([housing['SalePrice'], ensemble_pred],
                        axis = 1)

predictions.to_excel(predict_path + '1_Predictions.xlsx')
