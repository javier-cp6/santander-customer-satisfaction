import pandas as pd
import numpy as np
import xgboost as xgb

import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# Parameters

xgb_params = {
    "eta": 0.01,
    "max_depth": 6,
    "min_child_weight": 10,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 1,
}

output_file = f"model_xgb.bin"


# data preparation

df_full_train = pd.read_csv("data/train.csv")

del df_full_train["ID"]

cols = df_full_train.columns
constant_cols = [col for col in cols if df_full_train[col].std() == 0]
df_full_train.drop(constant_cols, axis=1, inplace=True)

duplicated_columns = []
cols = df_full_train.columns

for i, col in enumerate(cols[:-1]):
    for j in range(i + 1, len(cols)):
        if np.array_equal(df_full_train[col].values, df_full_train[cols[j]].values):
            duplicated_columns.append(cols[j])

df_full_train.drop(duplicated_columns, axis=1, inplace=True)

for df in [df_full_train]:
    df["var3"] = df["var3"].replace(to_replace=-999999, value=2)

df_train, df_val = train_test_split(df_full_train, test_size=0.20, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)


selected_features = [
    "imp_ent_var16_ult1",
    "imp_op_var39_comer_ult1",
    "imp_op_var39_comer_ult3",
    "imp_op_var39_efect_ult1",
    "imp_op_var39_efect_ult3",
    "imp_op_var39_ult1",
    "imp_op_var41_comer_ult1",
    "imp_op_var41_comer_ult3",
    "imp_op_var41_efect_ult3",
    "imp_op_var41_ult1",
    "imp_trans_var37_ult1",
    "imp_var43_emit_ult1",
    "ind_var39_0",
    "num_med_var22_ult3",
    "num_med_var45_ult3",
    "num_meses_var39_vig_ult3",
    "num_meses_var5_ult3",
    "num_op_var39_comer_ult1",
    "num_op_var39_comer_ult3",
    "num_op_var39_efect_ult3",
    "num_op_var39_hace2",
    "num_op_var39_ult1",
    "num_op_var39_ult3",
    "num_op_var41_comer_ult3",
    "num_op_var41_ult3",
    "num_var22_hace2",
    "num_var22_hace3",
    "num_var22_ult1",
    "num_var22_ult3",
    "num_var35",
    "num_var37_med_ult2",
    "num_var4",
    "num_var43_recib_ult1",
    "num_var45_hace2",
    "num_var45_hace3",
    "num_var45_ult1",
    "num_var45_ult3",
    "saldo_medio_var13_corto_hace2",
    "saldo_medio_var5_hace2",
    "saldo_medio_var5_hace3",
    "saldo_medio_var5_ult1",
    "saldo_medio_var5_ult3",
    "saldo_var25",
    "saldo_var26",
    "saldo_var30",
    "saldo_var37",
    "saldo_var42",
    "saldo_var5",
    "var15",
    "var3",
    "var36",
    "var38",
]


def train(df_train, y_train, xgb_params, num_boost_round=600):
    train_dicts = df_train[selected_features].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(train_dicts)

    features = list(dv.get_feature_names_out())

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

    model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)

    return dv, model


def predict(df_test, dv, model):
    test_dicts = df_test[selected_features].to_dict(orient="records")

    X_test = dv.transform(test_dicts)

    features = list(dv.get_feature_names_out())

    dtest = xgb.DMatrix(X_test, feature_names=features)

    y_pred = model.predict(dtest)

    return y_pred


# Training final model

print("Training final model")

dv, model = train(df_train, df_train["TARGET"].values, xgb_params, num_boost_round=600)

y_val = df_val["TARGET"].values

y_pred = predict(df_val, dv, model)

auc = roc_auc_score(y_val, y_pred)

print(f"auc={auc}")


# Save the model

with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"Model saved to {output_file}")
