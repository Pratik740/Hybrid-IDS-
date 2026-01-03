import pandas as pd
import numpy as np
import os

# corr_threshold = 0.9 

def correlation_feature_selection(
    csv_path,
    dataset_name,
    chunk_size=200_000
):
    reader = pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        on_bad_lines="skip"
    )

    corr_sum = None
    num_chunks = 0

    for chunk in reader:
        if "label" not in chunk.columns:
            continue

        X = chunk.drop(columns=["label"]).select_dtypes(include=[np.number]).astype("float32")  # coz iov2024 has non-numeric features (category,specific_class)

        corr = X.corr()

        if corr_sum is None:
            corr_sum = corr
        else:
            corr_sum += corr

        num_chunks += 1
        print(f"[{dataset_name}] Processed chunk {num_chunks}")

    
    corr_matrix = (corr_sum / num_chunks).abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        col for col in upper_triangle.columns
        if any(upper_triangle[col] > 0.9)
    ]

    all_features = pd.read_csv(
        csv_path,
        nrows=0
    ).columns.tolist()

    if "label" in all_features:
        all_features.remove("label")

    remaining_features = [f for f in all_features if f not in to_drop]

    selected_path = os.path.join(
        ".", f"{dataset_name}_corr_selected_features.txt"
    )

    pd.Series(remaining_features).to_csv(selected_path, index=False)

    print(f"Total chunks processed: {num_chunks}")
    print(f"Features before filtering: {corr_matrix.shape[0]}")
    print(f"Features removed: {len(to_drop)}")
    print(f"Features remaining: {len(remaining_features)}")
    print(f"Saved → {selected_path}")

    return remaining_features


# DDoS
correlation_feature_selection(
    csv_path="../data/balanced/ddos2019.csv",
    dataset_name="ddos"
)

# IoT
correlation_feature_selection(
    csv_path="../data/balanced/iot2023.csv",
    dataset_name="iot"
)

# IoV
correlation_feature_selection(
    csv_path="../data/balanced/iov2024.csv",
    dataset_name="iov"
)

