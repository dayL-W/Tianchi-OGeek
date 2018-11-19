import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

N = 5
threshold  = 0.5
n_folds = 5
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 28,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 1,
    'bagging_freq': 10,
    'verbose': 1
}

train_data = pd.read_csv('./cache/train_data.csv',index_col=0)
val_data = pd.read_csv('./cache/val_data.csv',index_col=0)
test_data = pd.read_csv('./cache/test_data.csv',index_col=0)

columns = train_data.columns
remove_columns = ["label",'prefix_title_tag_ctr', 'prefix_title_tag_ctrlog',
   'prefix_title_tag_click', 'prefix_title_tag_clicklog',
   'prefix_title_tag_count', 'prefix_title_tag_countlog',
    'max_similar','mean_similar','weight_similar']
# remove_columns = ['label']
features_columns = [column for column in columns if column not in remove_columns]

train_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
train_features = train_data[features_columns]
train_labels = train_data["label"]

val_data_length = val_data.shape[0]
validate_features = val_data[features_columns]
validate_labels = val_data["label"]
test_features = test_data[features_columns]
test_features = pd.concat([validate_features, test_features], axis=0, ignore_index=True)

print(train_features.shape)
print(test_features.shape)

kfolder = StratifiedKFold(n_splits=N, random_state=42, shuffle=True)
kfold = kfolder.split(train_features, train_labels)

preds_list = list()

for train_index, test_index in kfold:
    k_x_train = train_features.loc[train_index]
    k_y_train = train_labels.loc[train_index]
    k_x_test = train_features.loc[test_index]
    k_y_test = train_labels.loc[test_index]

    lgb_train = lgb.Dataset(k_x_train, k_y_train)
    lgb_eval = lgb.Dataset(k_x_test, k_y_test, reference=lgb_train)

    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=5000,
                    valid_sets=(lgb_train,lgb_eval),
                    early_stopping_rounds=50,
                    verbose_eval=50,
                    )
    test_preds = gbm.predict(test_features, num_iteration=gbm.best_iteration)
    preds_list.append(test_preds)

train_rate = sum(train_labels)/len(train_labels)
postive_num = int(train_rate*len(test_data))
length = len(preds_list)
preds_columns = ["preds_{id}".format(id=i) for i in range(length)]

preds_df = pd.DataFrame(data=preds_list)
preds_df = preds_df.T
preds_df.columns = preds_columns
preds_df = preds_df.copy()
preds_df["mean"] = preds_df.mean(axis=1)
validate_preds = preds_df[:val_data_length]
test_preds = preds_df[val_data_length:]

sort_preds = test_preds.sort_values('mean',ascending=False)
threshold = sort_preds.iloc[postive_num,-1]


preds_df["label"] = preds_df["mean"].apply(lambda item: 1 if item > threshold else 0)
validate_preds = preds_df[:val_data_length]
test_preds = preds_df[val_data_length:]
f_score = f1_score(validate_labels, validate_preds["label"])
print("The validate data's f1_score is {}".format(f_score))

predictions = pd.DataFrame({"predicted_score": test_preds["label"]})

print('postive num:',sum(test_preds['label']))
predictions.to_csv("./result.csv", index=False, header=False)

feat_imp = pd.Series(gbm.feature_importance(), index=features_columns).sort_values(ascending=False)
print('train :',sum(train_labels)/len(train_labels))
print('test :',sum(test_preds['label'])/len(test_preds['label']))