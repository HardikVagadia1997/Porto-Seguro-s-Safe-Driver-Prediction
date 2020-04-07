'''
###########################################################################################################################
-> FileName : Ensemble.py
-> Description : Creates Ensemble model
-> Author : Hardik Vagadia
-> E-Mail : vagadia49@gmail.com
-> Date : 27th March 2020
###########################################################################################################################
'''
#-------------------------------------------------------------------------------------------------------------------------
class Create_ensemble(object):
  def __init__(self, n_splits, base_models):
    self.n_splits = n_splits
    self.base_models = base_models
    
  def predict(self, X, y, T):
    X = np.array(X)
    y = np.array(y)
    T = np.array(T)

    folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=218).split(X, y))

    S_train = np.zeros((X.shape[0], len(self.base_models)))
    S_test = np.zeros((T.shape[0], len(self.base_models)))
        
    for i, clf in enumerate(self.base_models):
      S_test_i = np.zeros((T.shape[0], self.n_splits))
      for j, (train_idx, valid_idx) in enumerate(folds):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]
        clf.fit(X_train, y_train, verbose = 1)
        valid_pred = clf.predict_proba(X_valid)[:,1]
        S_train[valid_idx, i] = valid_pred
        S_test_i[:, j] = clf.predict_proba(T)[:,1]
            
      print( "\nTraining Gini for model {} : {}".format(i, u.eval_gini(y, S_train[:,i])))
      S_test[:, i] = S_test_i.mean(axis=1)
    return S_train, S_test