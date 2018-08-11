
# coding: utf-8

# # Machine Learning Challenge!

# ##  Mission
# - The objective is to devise the best possible model to predict successful/default loans using a preprocessed version of the Lending Club loan dataset.
# 
# 
# - The training data is 13689 loans labeled either as 1 (successful) or 0 (default). Comes with 30 categorical and numerical features. The testing data is also 13689 loans.
# 
# - Your profit will be determined by the amount of money you make from correctly predicting good loans (loan amount * interest rate/100.) subtracted by the money you lose from incorrectly predicting bad loans as good (-loan amount). I have given a function to calculate that.

# ### Online resources on Lending Club loan data
# Kaggle Page: https://www.kaggle.com/wendykan/lending-club-loan-data. Make sure to check out the kernels section.
# 
# Y Hat tutorial (It's in R, but its still useful): http://blog.yhat.com/posts/machine-learning-for-predicting-bad-loans.html
# 
# Blog tutorial on the data from Kevin Davenport: http://kldavenport.com/lending-club-data-analysis-revisted-with-python/
# 
# 

# In[3]:


from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

#Imports and set pandas options
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
pd.set_option("max.columns", 200)
pd.set_option("max.colwidth", 200)


# # Get the Data

# In[4]:


# Load in training data.
# Loan_status column is the target variable. Remember to drop it from df.
train = pd.read_csv("../../data/lending_club/challenge_training_data2.csv")
train.head(2)


# In[5]:


#Load in data dictionary
data_dict = pd.read_csv("../../data/lending_club/the_data_dictionary.csv")
data_dict.head(20)


# In[6]:


train.loan_status.value_counts()


# # Clean the Data and Get Correct Dtypes

# In[11]:


train.columns


# In[114]:


train["term"] = train.term.str.replace("months", "")
train["int_rate"] = train.int_rate.str.replace("%", "")
train["emp_length"] = train.emp_length.str.replace("years", "")
train["home_ownership"] = train.emp_length.str.replace("rent", "1")
train["home_ownership"] = train.emp_length.str.replace("own", "0")
train["verification_status"] = train.emp_length.str.replace("Not Verified", "0")
train["verification_status"] = train.emp_length.str.replace("Verified", "1")
train["verification_status"] = train.emp_length.str.replace("Source Verified", "1")


# In[138]:


train.term = train.term.astype(float)


# In[146]:


train.term = train.term.astype(float)
train["int_rate"] = train["int_rate"].astype(float)


# In[12]:


train["emp_length"][:10]


# In[13]:


train["int_rate"][:10]


# In[115]:


train.columns


# # Look at relationship between features

# In[131]:


train.corr().sort_values(by='loan_status', ascending = False).head(20)


# In[72]:


train.corr().sort_values(by='loan_status', ascending = False).head(5).index


# In[132]:


train.corr().sort_values(by='loan_status', ascending = True).head(20)


# In[116]:


train.corr().sort_values(by='loan_status', ascending = True).head(5).index


# # Feature Engineering

# In[117]:


train['fico_avg']= (train.fico_range_low +train.fico_range_low)/2
#train.drop(['fico_range_low','fico_range_low'], axis=1, inplace=True)


# # Making our X and target, leaving lots of stuff in for X and dummying a few features

# In[278]:


X = train[[u'fico_range_low', u'fico_range_high',
       u'annual_inc', 'fico_avg', 'int_rate'
           
, u'inq_last_6mths', u'revol_bal', u'pub_rec', u'dti', u'delinq_2yrs'
          ,"emp_length","home_ownership","verification_status"]]

X = X.fillna(0)
X = pd.get_dummies(X, columns=["emp_length","home_ownership","verification_status"], drop_first=True)

y = train['loan_status']


# In[ ]:


sum(X.columns == Xtest.columns ) == len(X.columns)


# In[152]:


X.head()


# # Using sklearn random forrest classifier

# In[98]:


# trying now with 500 . estimatores since this is usually effective in icnreasing model accuracy....
rfreg_gmv_tuned = RandomForestClassifier(n_estimators=300, 
                                #max_depth =20,
                                #max_features = 29,
                                random_state=1)

rfreg_gmv_tuned.fit(X, y)

cv = cross_val_score(rfreg_gmv_tuned, X, y, scoring='accuracy', cv=5)


# In[125]:


cv.mean()


# In[106]:


X.shape


# pd.DataFrame({'feature':X.columns,
#               'importance':rfreg_gmv_tuned.feature_importances_}).sort_values('importance', ascending = False)

# # Doing a random search here with the paramaters listed above

# In[119]:


estimator_range = range(100, 510, 50)
features_range = range(1, 9, 1)
depth_range = range(1, 5, 1)

param_grid_rf = {}
param_grid_rf["n_estimators"] = estimator_range
param_grid_rf["max_features"] = features_range
param_grid_rf["max_depth"] = depth_range


# In[120]:


rand_grid_rf_2 = RandomizedSearchCV(estimator = RandomForestClassifier(), n_iter = 15,
                        param_distributions = param_grid_rf, cv = 4, scoring='accuracy')


#fit rand search
rand_grid_rf_2.fit(X, y)


# In[122]:


rand_grid_rf_2.best_params_


# In[123]:


rand_grid_rf_2.best_score_


# In[124]:


X.head()


# # taking the best paramaters and making a model

# In[153]:




# trying now with 500 . estimatores since this is usually effective in icnreasing model accuracy....
rfreg_gmv_tuned = RandomForestClassifier(n_estimators=300, 
                                max_depth =4,
                                max_features = 4,
                                random_state=1)

rfreg_gmv_tuned.fit(X, y)

cv = cross_val_score(rfreg_gmv_tuned, X, y, scoring='accuracy', cv=5)


# In[154]:


cv.mean()


# In[156]:


X.head()


# In[157]:


rfreg_gmv_tuned = RandomForestClassifier(n_estimators=300, 
                                random_state=1)

rfreg_gmv_tuned.fit(X, y)

cv = cross_val_score(rfreg_gmv_tuned, X, y, scoring='accuracy', cv=5)
cv.mean()


# In[159]:


cv.mean()


# In[158]:


pd.DataFrame({'feature':X.columns,
              'importance':rfreg_gmv_tuned.feature_importances_}
            ).sort_values('importance', ascending = False).head(10)


# In[160]:


rfreg_gmv_tuned = RandomForestClassifier(n_estimators=300, 
                                random_state=1)

rfreg_gmv_tuned.fit(X, y)

cv = cross_val_score(rfreg_gmv_tuned, X, y, scoring='precision', cv=5)


# In[161]:


cv.mean()


# In[167]:


estimator_range = range(200, 510, 50)
features_range = range(1, 30, 3)
depth_range = range(1, 15, 2)

param_grid_rf = {}
param_grid_rf["n_estimators"] = estimator_range
param_grid_rf["max_features"] = features_range
param_grid_rf["max_depth"] = depth_range


# In[168]:


estimator_range


# In[170]:


estimator_range = range(200, 510, 50)
features_range = range(1, 30, 3)
depth_range = range(1, 15, 2)

param_grid_rf = {}
param_grid_rf["n_estimators"] = estimator_range
param_grid_rf["max_features"] = features_range
param_grid_rf["max_depth"] = depth_range


rand_grid_rf_2 = RandomizedSearchCV(estimator = RandomForestClassifier(), n_iter =30,
                        param_distributions = param_grid_rf, cv = 5, scoring='precision')


#fit rand search
rand_grid_rf_2.fit(X, y)

rand_grid_rf_2.best_params_


# In[171]:


rand_grid_rf_2.best_params_


# In[172]:


rand_grid_rf_2.best_score_


# # After testing some different models, this is the final

# In[173]:


rfreg_gmv_tuned = RandomForestClassifier(n_estimators=450, 
                                max_depth =13,
                                max_features = 28,
                                random_state=1)

rfreg_gmv_tuned.fit(X, y)

cv = cross_val_score(rfreg_gmv_tuned, X, y, scoring='accuracy', cv=5)


# In[174]:


cv.mean()


# In[335]:


pd.DataFrame({'feature':X.columns,
              'importance':rfreg_gmv_tuned.feature_importances_}
            ).sort_values('importance', ascending = False).head(10)


# In[339]:


# same thing here but I graphed it out so it was a little easier to visualize 
pd.Series(rfreg_gmv_tuned.feature_importances_, 
          index=X.columns).sort_values().plot(kind='barh', figsize=(13,50))


# In[336]:


X.columns


# In[337]:


X.head()


# # Now pulling in the testing dataset

# In[175]:


#Load in test data when you're ready 
test = pd.read_csv("../../data/lending_club/challenge_testing_data.csv")


# # Formatting test data in same way as the training set

# In[176]:


test["term"] = test.term.str.replace("months", "")
test["int_rate"] = test.int_rate.str.replace("%", "")
test["emp_length"] = test.emp_length.str.replace("years", "")
test["home_ownership"] = test.emp_length.str.replace("rent", "1")
test["home_ownership"] = test.emp_length.str.replace("own", "0")
test["verification_status"] = test.emp_length.str.replace("Not Verified", "0")
test["verification_status"] = test.emp_length.str.replace("Verified", "1")
test["verification_status"] = test.emp_length.str.replace("Source Verified", "1")


# In[177]:


test.term = test.term.astype(float)
test["int_rate"] = test["int_rate"].astype(float)


# In[178]:


test['fico_avg']= (test.fico_range_low +test.fico_range_low)/2


# In[179]:


X = test[[u'fico_range_low', u'fico_range_high',
       u'annual_inc', 'fico_avg', 'int_rate'
, u'inq_last_6mths', u'revol_bal', u'pub_rec', u'dti', u'delinq_2yrs'
          ,"emp_length","home_ownership","verification_status"]]

X = X.fillna(0)
X = pd.get_dummies(X, columns=["emp_length","home_ownership","verification_status"], drop_first=True)

y = test['loan_status']


# In[ ]:


X.co


# In[180]:


X.head()


# # Scoring the model and adding it to the dataframe

# In[191]:


test['model_scored'] = pd.Series(list(rfreg_gmv_tuned.predict_proba(X)))


# In[198]:


test['model_scored'] = rfreg_gmv_tuned.predict_proba(X)[:, 1]


# In[184]:


test.model_scored.value_counts(normalize=True)


# In[195]:


test.head()


# # Functions below to determine the threshold for loan approval

# In[186]:


def threshold(probs, thres = 0.5):
    output = np.where(probs >= thres, 1, 0)
    return output
def profit_function(data):
    if data.target == 0 and data.predicted == 1:
        return -1 *data.loan_amnt
    elif data.target == 1 and data.predicted == 1:
        return data.loan_amnt * (data.int_rate/100.)
    else:
        return 0


# In[ ]:


#List of probabilities
probs = np.array([0.2, 0.5, 0.8, 0.9, 0.1, 0.75])
#Pass in probabilities into threshold function, using .7 as threshold
preds = threshold(probs, thres=.7)
preds
sample_df = {"loan_amnt": [1000, 500, 200, 5000, 3000, 6000],
            "int_rate": [18, 20, 4, 5, 2, 10], 
            "target": [0, 1, 0, 1, 0, 1]}
#Put dictionary in data frame
profit_df = train[['loan_amnt','int_rate','loan_status']


# Here is the threshold function. Input your probabilities for class 1 and set a probability threshold of your choice. The default threshold is 0.5. The output will be 1's and 0s, 1 values for all the values that are greater or equal to your predetermined threshold value.

# In[199]:



def threshold(probs, thres = 0.5):
    output = np.where(probs >= thres, 1, 0)
    return output


# This is the profit function. It takes in a dataframe with the loan_amnt, int_rate, target variable, and class predictions values.
# 
# - It first checks to see if a row has 0 in the outcome column and 1 in the predicted (false positive) and returns the negative value of the loan_amnt. This is how much money you lost for loans that mean that condition.
# 
# - Then it checks for true positives, meaning conditions where both the target and predicted values equal 1, then return the loan_amnt times the int_rate divided by 100. This is how much money you made from loans that meet this condition.
# 
# - Everything else gets a zero.

# In[221]:


def profit_function(data):
    if data.loan_status == 0 and data.predicted == 1:
        return -1 *data.loan_amnt
    elif data.loan_status == 1 and data.predicted == 1:
        return data.loan_amnt * (data.int_rate/100.)
    else:
        return 0


# Here's an example of using the threshold and profit_functions

# In[279]:


df_send = test[['loan_amnt', 'int_rate', 'loan_status', 'model_scored']]


# In[280]:


df_send.head()


# In[325]:


#List of probabilities
probs = np.array([0.2, 0.5, 0.8, 0.9, 0.1, 0.75])

#Pass in probabilities into threshold function, using .7 as threshold
preds = threshold(test.model_scored, thres=.5)
preds


# In[326]:


test_df = test[['loan_amnt', 'int_rate', 'loan_status']]

#Put dictionary in data frame

profit_df = pd.DataFrame(test_df)

profit_df


# In[327]:


#Add in predictions

profit_df["predicted"] = preds
profit_df


# # Here is how much money we make or loose on each loan depending on the predicted probability and the threshold we set

# In[329]:


#Apply function onto data frame
profit_series = profit_df.apply(profit_function, axis = 1)
profit_series.head()


# In[330]:


#Sum up profits and losses
profit_series.sum()


# This model made $650

# In[331]:


feature_range = range(80, 100, 1)


# In[3]:


#feature_range = (feature_range.atype(int)) /100


# # Here is a for loop so we can find the profitabilty for different thresholds on top of our predicted probabilites (from the model). This will give us the optimal proability threshold we should apply on top our trained model

# In[334]:


df_send.columns

thresh = range(10)
t = .9290

for x in thresh:
    preds = threshold(df_send.model_scored, thres=t)
    profit_df["predicted"] = preds
    profit_series = profit_df.apply(profit_function, axis = 1)
    print(str(t)+":"+str(profit_series.sum()))
    t=t+.0001


# # Final results is a threshold of .9298 and we will make $681,160.53 in profit! This means we only aprove loans that have a very high confidence level
