import pandas as pd
from lstm_model import build_cascaded_lstm
from data_utils import prepare_input
from train_hybrid import train_hybrid

def load_sfs_filtered_df(csv_path, feature_txt_path):
    # Load selected features
    selected_features = (
        pd.read_csv(feature_txt_path, header=None)[0]
        .astype(str)
        .str.strip()
        .tolist()
    )

    selected_features = [
        f for f in selected_features
        if f != "0" and f != "label"
    ]

    usecols = selected_features + ["label"]

    # Load only required columns
    df = pd.read_csv(csv_path, usecols=usecols)

    return df



datasets = {
    "iot_2023": {
        "csv": "../data/balanced/iot2023.csv",
        "features": "../feature_selection/iot_sfs_selected.txt"
    },
    # "iov_2024": {
    #     "csv": "../data/balanced/iov2024.csv",
    #     "features": "../feature_selection/iov_sfs_selected.txt"
    # },
    # "ddos_2019": {
    #     "csv": "data/balanced/ddos2019.csv",
    #     "features": "feature_selection/ddos_sfs_selected.txt"
    # }
}

for name, paths in datasets.items():

    print(f"\nTraining hybrid model for {name}")

    df = load_sfs_filtered_df(
        csv_path=paths["csv"],
        feature_txt_path=paths["features"]
    )

    print(df.shape)
    print(df.columns)

    train_hybrid(
        df=df,
        model_builder={
            "model": build_cascaded_lstm,
            "prepare": prepare_input
        },
        save_prefix=name
    )
