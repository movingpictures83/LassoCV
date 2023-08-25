#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[328]:


################################################ Preprocessing #########################################################
class LassoCVPlugin:
 def input(self, inputfile):
  self.data_path = inputfile

 def run(self):
  pass

 def output(self, outputfile):
  #categorical_cols = ["Race_Ethnicity"]


  data_df = pd.read_csv(self.data_path)

  # # Tramsform categorical data to categorical format:
  # for category in categorical_cols:
  #     data_df[category] = data_df[category].astype('category')
  #

  # Clean numbers:
  #"Cocain_Use": {"yes":1, "no":0},
  cleanup_nums = { "Cocain_Use": {"yes":1, "no":0},
                 "race": {"White":1, "Black":0, "BlackIsraelite":0, "Latina":1},
  }

  data_df.replace(cleanup_nums, inplace=True)

  # Drop id column:
  data_df = data_df.drop(["pilotpid"], axis=1)

  # remove NaN:
  data_df = data_df.fillna(0)

  # Standartize variables
  from sklearn import preprocessing
  names = data_df.columns
  scaler = preprocessing.StandardScaler()
  data_df_scaled = scaler.fit_transform(data_df)
  data_df_scaled = pd.DataFrame(data_df_scaled, columns=names)


  # In[303]:


  ################################################ Inflamation marker #########################################################

  # Split training and test
  # Scale training

  y_col = "interleukin6"
  test_size = 0.25
  validate = True
  random_state = 22

  y = data_df[y_col]

  X = data_df_scaled.drop([y_col], axis=1)

  # Create random variable for benchmarking
  #X["random"] = np.random.random(size= len(X))




  from sklearn.linear_model import LassoCV
  from sklearn.feature_selection import SelectFromModel
  from sklearn.linear_model import LogisticRegression


  #clf = LassoCV(cv=5)

  model_lasso = LassoCV(alphas = [10, 5, 1, 0.1, 0.001, 0.0005], cv=5).fit(X, y)

  coef = pd.Series(model_lasso.coef_, index = X.columns)
  #coef_pd = pd.DataFrame({"coef:"coef, "features":coef.index})
  print(coef.head())
  print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


  # In[292]:


  # # F-test - can't apply because of negative values

  # from sklearn.feature_selection import SelectKBest
  # from sklearn.feature_selection import chi2

  # # #
  # #apply SelectKBest class to extract top 10 best features
  # #
  # bestfeatures = SelectKBest(score_func=chi2, k=15)
  # fit = bestfeatures.fit(X_train_transformed,y_train)
  # # dfscores = pd.DataFrame(fit.scores_)
  # # dfcolumns = pd.DataFrame(X.columns)
  # # #concat two dataframes for better visualization
  # # featureScores = pd.concat([dfcolumns,dfscores],axis=1)
  # # featureScores.columns = ['Specs','Score']  #naming the dataframe columns
  # # featureScores = featureScores[featureScores["Score"]>1]
  # # featureScores = featureScores.sort_values(by="Score")
  # # print("number of features: {}".format(len(featureScores)))
  # # important_features = list(featureScores["Specs"])

# #

