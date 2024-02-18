
import numpy as np 
import polars as pl
from sklearn.metrics import roc_auc_score 

def get_feature_definitions(columns, df_feature_definition):
    return pl.DataFrame({'Variable': columns}).join(
        df_feature_definition,
        on = 'Variable',
        how = 'left',
    )

def gini_stability(base, w_fallingrate=88.0, w_resstd=-0.5):
    gini_in_time = base.loc[:, ["WEEK_NUM", "target", "score"]]\
        .sort_values("WEEK_NUM")\
        .groupby("WEEK_NUM")[["target", "score"]]\
        .apply(lambda x: 2*roc_auc_score(x["target"], x["score"])-1).tolist()
    
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std


def convert_type(df):
    for col in df.columns:
        if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
            df = df.with_columns(pl.col(col).cast(pl.Int64));
        elif col in ["date_decision"] or col[-1] in ("D",):
            df = df.with_columns(pl.col(col).cast(pl.Date));
        elif col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64));
        elif col[-1] in ("M",):
            df = df.with_columns(pl.col(col).cast(pl.String));
    return df;

def extract_columns_tipe(df):
    features_num = list(df.select_dtypes('number'))
    features_total = df.columns.tolist()
    features_date = [el for el in features_total if el.endswith("D")]
    features_cat = [el for el in features_total if el not in (features_num + features_date)]
    features_num.remove('case_id')
    if "num_group1" in features_num:
        features_num.remove("num_group1")
    if "num_group2" in features_num:
        features_num.remove("num_group2")
    return features_num, features_date, features_cat 

def aggregate_num_features_by_historic(df, col_list, col_sort):
    operation_to_apply = []
    for col in col_list:
        operation_to_apply.append(pl.col(col).mean().alias(f"{col}_mean"))
        operation_to_apply.append(pl.col(col).std().alias(f"{col}_std"))
        operation_to_apply.append(pl.col(col).last().alias(f"{col}_last"))
        operation_to_apply.append(pl.col(col).last().alias(f"{col}_first"))
    df_grouped = df.sort(by=col_sort, descending=True).group_by("case_id").agg(operation_to_apply)
    return df_grouped

