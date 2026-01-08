import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def run_sfs_xgboost(
    csv_path,
    feature_list_path,
    dataset_name="iot",
    max_samples=200_000,   # key safety parameter
):

    selected_features = (
        pd.read_csv(feature_list_path, header=None)[0]
        .astype(str)
        .str.strip()
        .tolist()
    )

    print(f"Features after correlation: {len(selected_features)}")

    usecols = selected_features + ["label"]

    df = pd.read_csv(
        csv_path,
        usecols=usecols,
        on_bad_lines="skip"
    )

    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Stratified subsampling
    if len(df) > max_samples:
        print(f"Subsampling from {len(df)} → {max_samples} rows (stratified)")

        n_classes = df["label"].nunique()
        samples_per_class = max_samples // n_classes

        df = (
            df.groupby("label", group_keys=False)
              .apply(lambda x: x.sample(
                  n=min(len(x), samples_per_class),
                  random_state=42
              ))
              .reset_index(drop=True)
        )

    X = df[selected_features].select_dtypes(include=[np.number])  
    y = df["label"]
    
    print("Pre 1")

    print("Feature matrix:", X.shape)
    print("Label distribution:")
    print(y.value_counts())

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("xgb", XGBClassifier(
            n_estimators=50,      
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=1                # Set to 1 to avoid OS kill, cpu was killing process :(
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)

    sfs = SequentialFeatureSelector(
        estimator=pipeline,
        k_features="best",
        forward=False,          # backward elimination
        floating=False,
        scoring="accuracy",
        cv=cv,
        n_jobs=1                
    )
    print("Started")

    sfs.fit(X, y)
    final_features = list(sfs.k_feature_names_)
    print(f"Final selected features: {len(final_features)}")
    print("Sample:", final_features[:10])

    output_path = f"{dataset_name}_sfs_selected_features.txt"
    pd.Series(final_features).to_csv(output_path, index=False)

    print(f"Saved → {output_path}")


# run_sfs_xgboost(
#     csv_path="../data/balanced/iot2023.csv",
#     feature_list_path="./iot_corr_selected.txt",
#     dataset_name="iot",
# )
# Output:
# Final 20 features were selected for iot2023


# run_sfs_xgboost(
#         csv_path="../data/balanced/iov2024.csv",
#         feature_list_path="./iov_corr_selected.txt",
#         dataset_name="iov",
#     )
# Output:
# Final 7 features were selected for iov2024


run_sfs_xgboost(
        csv_path="../data/balanced/ddos2019.csv",
        feature_list_path="./ddos_corr_selected.txt",
        dataset_name="ddos",
    )

