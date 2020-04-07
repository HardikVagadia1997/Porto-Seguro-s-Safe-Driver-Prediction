'''
###########################################################################################################################
-> FileName : FeatureEngg.py
-> Description : This file contains code for feature engineering. It transforms raw data into a variety of different useful features.
-> Author : Hardik Vagadia
-> E-Mail : vagadia49@gmail.com
-> Date : 27th March 2020
###########################################################################################################################
'''

#-------------------------------------------------------------------------------------------------------------------------

#Importing essential libraries
import warnings
warnings.filterwarnings('ignore')
from Modules import utils as u
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

#-------------------------------------------------------------------------------------------------------------------------


def feature_engineering(data_tr, data_te) :
    '''
    -> This function performs different feature transformations
       and saves them in .csv files
    '''
    #Feature Engineering
    
    #Loading data
    data_tr = pd.read_csv('train.csv')
    data_te = pd.read_csv('test.csv')
    id_tr = data_tr["id"]
    id_te = data_te["id"]
    y_tr = data_tr["target"]
    data_tr = data_tr.drop(["id", "target"], axis = 1)
    data_te = data_te.drop("id", axis = 1)
    
    #Dropping 'calc' features
    #We are dropping these features as they do not show any significant
    #impact on the target variables.
    calc_features = []
    for f in data_tr.columns :
        if 'calc' in f :
            calc_features.append(f)
    data_tr = data_tr.drop(calc_features, axis = 1)
    data_te = data_te.drop(calc_features, axis = 1)

    
    #Dividing features into different categories
    categorical_features = [x for x in data_tr.columns if 'cat' in x]
    binary_features = [x for x in data_tr.columns if 'bin' in x]
    numerical_features = [x for x in data_tr.columns if ('cat' not in x and 'bin' not in x)]
    
    #Adding new feature containing number of missing Values in each row
    data_tr['missing'], data_te['missing'] = u.missing_data(data_tr, data_te)
    
    #Multiply and divide the continuous features with each-other and create some new features
    conti_features = ['ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']
    md_features = []
    for idx, (x,y) in enumerate(combinations(conti_features, 2)) :
        data_tr, data_te, features = u.mul_div(data_tr, data_te, x, y, idx)
        md_features.extend(features)
    #Applying column-standardization on the multiply-divide features
    data_tr, data_te = u.col_std(features, data_tr, data_te)
    
    #String features
    category_list = ['ind', 'reg', 'car']
    data_tr, data_te, str_features = u.string_feature(data_tr, data_te, category_list)
    #Applying Ordinal Encoding to String Features
    data_tr, data_te = u.ordinal_enc(str_features, data_tr, data_te)

    #Categorical count features
    cat_count_features = categorical_features + ['new_ind', 'new_reg', 'new_car']
    data_tr, data_te, cat_count_feature_names = u.cat_cnt(data_tr, data_te, cat_count_features)
    #Applying column-standardization on the categorical-count features
    data_tr, data_te = u.col_std(cat_count_feature_names, data_tr, data_te)
    
    #Sum of all the binary features in a row
    data_tr = u.sum_feature("sum_all_bin", data_tr, binary_features)
    data_te = u.sum_feature("sum_all_bin", data_te, binary_features)
    
    #Absolute difference of each row from the reference row
    data_tr = u.abs_diff_sum("bin_diff", data_tr, binary_features)
    data_te = u.abs_diff_sum("bin_diff", data_te, binary_features)
    
    #Group Statistical Features
    target_features = ['ps_ind_01', 'ps_ind_03', 'ps_ind_15','ps_car_01_cat', 'ps_car_06_cat', 'ps_car_11_cat']
    group_features = ['ps_ind_01', 'ps_ind_03', 'ps_ind_15','ps_car_01_cat', 'ps_car_06_cat', 'ps_car_11_cat']
    data_tr, size, mean, std, median, Max, Min = u.group_stat_features(data_tr, group_features, target_features)
    data_te, _, _, _, _, _, _ = u.group_stat_features(data_te, group_features, target_features)
    
    #In categorical features, some values are -1
    #We will apply OrdinalEncoding on the categorical features
    data_tr, data_te = u.ordinal_enc(categorical_features + 'ps_car_11', data_tr, data_te)
    
    #Applying StandardScaler on some continuous features
    conti_features = ['ps_reg_02', 'ps_reg_03', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']
    data_tr, data_te = u.col_std(conti_features, data_tr, data_te)
    
    x_tr = pd.concat([id_tr, data_tr], axis = 1)
    x_te = pd.concat([id_te, data_te], axis = 1)
    #Saving feature engineered data to .csv files for future use
    x_tr.to_csv('final_train.csv')
    x_te.to_csv('final_test.csv')
    y_tr.to_csv('target_labels.csv')