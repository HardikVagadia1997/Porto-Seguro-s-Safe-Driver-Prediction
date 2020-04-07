'''
###########################################################################################################################
-> FileName : utils.py
-> Description : This file contains all the important functions used throughout this project
-> Author : Hardik Vagadia
-> E-Mail : vagadia49@gmail.com
-> Date : 27th March 2020
###########################################################################################################################
'''
#-------------------------------------------------------------------------------------------------------------------------
def eval_gini(y_true, y_prob):
	'''
	-> Source : https://www.kaggle.com/mashavasilenko/porto-seguro-xgb-modeling-and-parameters-tuning
	-> This function returns gini score by taking actual labels and predicted probability as input
	'''
	y_true = np.asarray(y_true)
	y_true = y_true[np.argsort(y_prob)]
	ntrue = 0
	gini = 0
	delta = 0
	n = len(y_true)
	for i in range(n-1, -1, -1):
		y_i = y_true[i]
		ntrue += y_i
		gini += y_i * delta
		delta += 1 - y_i
	gini = 1 - 2 * gini / (ntrue * (n - ntrue))
	return gini

#-------------------------------------------------------------------------------------------------------------------------

def gini_xgb(preds, dtrain):
	labels = dtrain.get_label()
	gini_score = -eval_gini(labels, preds)
	return [('gini', gini_score)]

#-------------------------------------------------------------------------------------------------------------------------

def gini_normalized(a, p):
	'''
	-> Returns normalized gini impurity
	'''
	return gini(a, p) / gini(a, a)

#-------------------------------------------------------------------------------------------------------------------------

def missing_data(tr, te) :
	"""
	-> Function to calculate new feature that counts number of missing values per row
	-> These are discrete values b/w 0 and 8
	-> No need to perform any sort of encoding on this feature
	"""
	tr['missing'] = (tr == -1).sum(axis = 1).astype('float')
	te['missing'] = (te == -1).sum(axis = 1).astype('float')
	return tr['missing'], te['missing']

#-------------------------------------------------------------------------------------------------------------------------

def mul_div(tr, te, f1, f2, idx) :
	"""
	-> This function will multiply and divide the given features with each other and add them as new features
	-> Multiply and divide the continuous features with each-other and create some new features
	-> These features are continuous values and hence can be treated as numerical features
	-> We will apply StandardScaler on these feature
	"""
	tr['mul_{}'.format(idx)] = tr[f1]*tr[f2]
	tr['div_{}'.format(idx)] = tr[f1]/(tr[f2] + 1.001) #Adding 1.001 in denominator to prevent division by zero
	te['mul_{}'.format(idx)] = te[f1]*te[f2]
	te['div_{}'.format(idx)] = te[f1]/(te[f2] + 1.001) #Adding 1.001 in denominator to prevent division by zero
	features = ['mul_{}'.format(idx), 'div_{}'.format(idx)]
	return tr, te, features

#-------------------------------------------------------------------------------------------------------------------------

def string_feature(tr, te, category_list) :
  """
  -> Function to get the string features
  -> These features will be treated as categorical features and will apply ordinal encoding on this feature
  """
  feature_names_str = []
  for cat in tqdm(category_list) :
  	feature_names = list(tr.columns)
  	features = [c for c in feature_names if cat in c]
  	name = 'new_' + cat 	#New feature name
  	feature_names_str.append(name)
  	#String features for train data
  	cnt = 0
  	for c in features :
  		if cnt == 0:
  			tr[name] = tr[c].astype(str)
  			cnt+=1
  		else :
  			tr[name] += '_'+tr[c].astype(str)
  	#String features for test data
  	cnt = 0
  	for c in features :
  		if cnt == 0:
  			te[name] = te[c].astype(str)
  			cnt+=1
  		else :
  			te[name] += '_'+te[c].astype(str)
  return tr, te, feature_names_str

#-------------------------------------------------------------------------------------------------------------------------

def cat_cnt(tr, te, cat_count_features) :
  """
  -> Categorical count features
  -> This feature will be treated as a numerical feature
  -> We will apply StandardScaler on top of it
  """
  feature_names = []
  for c in tqdm(cat_count_features) :
	  d = pd.concat([tr[c],te[c]]).value_counts().to_dict()
	  tr['%s_count'%c] = tr[c].apply(lambda x:d.get(x,0))
	  te['%s_count'%c] = te[c].apply(lambda x:d.get(x,0))
	  feature_names.append('%s_count'%c)
  return tr, te, feature_names

#-------------------------------------------------------------------------------------------------------------------------

def sum_feature(new_feature_name, data, features) :
	"""
	-> Sum of all the binary features in a row
	-> This feature contains discrete values b/w 0 and 6
	-> No need for any kind of encoding
	"""
	data[new_feature_name] = data[features].sum(axis = 1)
	return data

#-------------------------------------------------------------------------------------------------------------------------

def abs_diff_sum(new_feature_name, data, features) :
	"""
	-> This function measures how different certain binary observations are.
	-> Absolute difference of each row from the reference row
	-> Discrete value feature
	-> No need for any sort of encoding
	"""
	ref_row = data[features].median(axis = 0)
	ref_row = list(map(int, ref_row.to_list()))
	data[new_feature_name] = abs(data[features] - ref_row).sum(axis = 1)
	return data

#-------------------------------------------------------------------------------------------------------------------------

def group_stat_features(data, group_features, target_features) :
	"""
	-> Group Statistical Features
	-> These are discrete numerical values
	-> No encoding
	"""
	size_features = []
	mean_features = []
	Max_features = []
	Min_features = []
	std_features = []
	median_features = []
	for t in tqdm(target_features) :
		for g in group_features :
			if t != g :
				temp_df = data[[g, t]]
				
				#Size features
				size = pd.DataFrame(temp_df.groupby(g).size()).reset_index()
				size.columns = [g, '%s_size' %t]
				size_features.append('%s_size' %t)
				data = data.merge(size, on=g, how = 'left')
				del size
				
				#Mean features
				mean = pd.DataFrame(temp_df.groupby(g).mean()).reset_index()
				mean.columns = [g, '%s_mean' %t]
				mean_features.append('%s_mean' %t)
				data = data.merge(mean, on=g, how = 'left')
				del mean
				
				#Std-Dev features
				std = pd.DataFrame(temp_df.groupby(g).std()).reset_index()
				std.columns = [g, '%s_std_dev' %t]
				std_features.append('%s_std_dev' %t)
				data = data.merge(std, on=g, how = 'left')
				del std
				
				#Median features
				median = pd.DataFrame(temp_df.groupby(g).median()).reset_index()
				median.columns = [g, '%s_median' %t]
				median_features.append('%s_median' %t)
				data = data.merge(median, on=g, how = 'left')
				del median
				
				#Max features
				Max = pd.DataFrame(temp_df.groupby(g).max()).reset_index()
				Max.columns = [g, '%s_Max' %t]
				Max_features.append('%s_Max' %t)
				data = data.merge(Max, on=g, how = 'left')
				del Max
				
				#Min features
				Min = pd.DataFrame(temp_df.groupby(g).min()).reset_index()
				Min.columns = [g, '%s_Min' %t]
				Min_features.append('%s_Min' %t)
				data = data.merge(Min, on=g, how = 'left')
				del Min
	return data, size_features, mean_features, std_features, median_features, Max_features, Min_features

#-------------------------------------------------------------------------------------------------------------------------

def col_std(feature_names, data_tr, data_te) :
	'''
	-> Applies column stadardization on numerical features
	   and returns data
	'''
	for features in tqdm(feature_names) :
		std = StandardScaler()
		std.fit(data_tr[feature].values.reshape(-1,1))
		data_tr[feature] = std.transform(data_tr[feature].values.reshape(-1,1))
		data_te[feature] = std.transform(data_te[feature].values.reshape(-1,1))
	return data_tr, data_te

#-------------------------------------------------------------------------------------------------------------------------

def ordinal_enc(feature_names, data_tr, data_te) :
	'''
	-> Applies Ordinal Encoding on Categorical features
       and returns data
	'''
	for f in tqdm(categorical_features) :
		ord_enc = OrdinalEncoder()
		ord_enc.fit(data_tr[f].values.reshape(-1,1))
		data_tr[f] = ord_enc.transform(data_tr[f].values.reshape(-1,1))
		data_te[f] = ord_enc.transform(data_te[f].values.reshape(-1,1))
	return data_tr, data_te

#-------------------------------------------------------------------------------------------------------------------------