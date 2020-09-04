import pandas as pd
from collections import Counter
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import lightgbm as lgbm
import xgboost as xgb

# Load data sets
train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("test.csv")
submission_df = pd.read_csv("sample_submission.csv")

# Create a merged data set and review initial information
combined_df = pd.concat([train_df, val_df])
print(combined_df.describe())

# Check missing values
print(combined_df.columns[combined_df.isnull().any()])

# Get percentage of missing values for cols with missing values
print(combined_df[[]].isnull().sum()/len(combined_df)*100)

# Get the data types
print(Counter([combined_df[col].dtype for col in combined_df.columns.values.tolist()]).items())

# Set the ID col as index
combined_df.set_index('id', inplace = True)

# Create dummies for categorical and binary values (For LGBM train remove it)
#combined_df = pd.get_dummies(combined_df, columns = [col for col in combined_df if col.endswith('bin') or col.endswith('cat')])

# Get the data types again to check our transformation
print(Counter([combined_df[col].dtype for col in combined_df.columns.values.tolist()]).items())

# Split combined_df into train_df and val_df again
train_df = combined_df.loc[combined_df["target"].isin([1, 0])]
val_df = combined_df[combined_df.index.isin(submission_df["id"])]
val_df = val_df.drop(["target"], axis = 1)

# Create X_train_df and y_train_df set
X_train_df = train_df.drop("target", axis = 1)
y_train_df = train_df["target"]




# SCALE

# Scale the data and use RobustScaler to minimise the effect of outliers
#scaler = RobustScaler()
#scaler = MinMaxScaler()
scaler = StandardScaler()

# Scale the X_train set
X_train_scaled = scaler.fit_transform(X_train_df.values)
X_train_df = pd.DataFrame(X_train_scaled, index = X_train_df.index, columns= X_train_df.columns)

# Scale the X_test set
val_scaled = scaler.transform(val_df.values)
val_df = pd.DataFrame(val_scaled, index = val_df.index, columns= val_df.columns)




# TRAIN TEST SPLIT

# Split our training sample into train and test, leave 20% for test
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state = 20)






# CLASS IMBALANCE

# Upsample minority class

# Concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# Separate minority and majority classes
not_target = X[X.target==0]
yes_target = X[X.target==1]

not_target_up = resample(yes_target,
                                replace = True, # sample without replacement
                                n_samples = len(not_target), # match majority n
                                random_state = 27) # reproducible results

# Combine minority and downsampled majority
upsampled = pd.concat([not_target_up, not_target])

# Checking counts
print(upsampled.target.value_counts())

# Create training set again
y_train = upsampled.target
X_train = upsampled.drop('target', axis=1)

print(len(X_train))




# MODEL

# LIGHT GBM

# Indicate the categorical features for the LGBM classifier
categorical_features = [c for c, col in enumerate(X_train.columns) if col.endswith('cat')]

# Get the train and test data for the training sequence
train_data = lgbm.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
test_data = lgbm.Dataset(X_test, label=y_test)


parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    #'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

classifier = lgbm.train(parameters,
                       train_data,
                       valid_sets= test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# Make predictions
predictions = classifier.predict(val_df.values)

# Create submission file
my_pred_lgbm = pd.DataFrame({'id': val_df.index, 'target': predictions})

# Create CSV file
my_pred_lgbm.to_csv('pred_lgbm.csv', index=False)




"""

# XGBoost

# Get the train and test data for the training sequence
train_data = xgb.DMatrix(X_train, label=y_train)
test_data = xgb.DMatrix(X_test, label=y_test)


parameters = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9,'learning_rate':0.05,
          'objective': 'binary:logistic', 'eval_metric': 'auc', 'silent': True}

classifier = xgb.train(parameters,
                       train_data,
                       evals=[(train_data, "X_train"),(test_data, "X_test")],
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# Make predictions
predictions = classifier.predict(xgb.DMatrix(val_df))

# Create submission file
my_pred_xgb = pd.DataFrame({'id': val_df.index, 'target': predictions})

# Create CSV file
my_pred_xgb.to_csv('pred_xgb.csv', index=False)



"""



