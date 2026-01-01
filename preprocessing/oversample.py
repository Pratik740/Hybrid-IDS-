from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import os 
import gc


def oversample(file_path, out_path):
    print("Processing :", file_path)

    # Read normally
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    # Keep only numeric columns
    df = df.select_dtypes(include=["number"])

    if "label" not in df.columns:
        raise ValueError("Label column missing after numeric filtering")

    # Convert to float32 AFTER filtering
    df = df.astype("float32")

    X = df.drop(columns=["label"])
    y = df["label"]

    del df
    gc.collect()

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    del X, y
    gc.collect()

    df_balanced = X_res.copy()
    df_balanced["label"] = y_res

    print(df_balanced["label"].value_counts())

    df_balanced.to_csv(out_path, index=False)

    del df_balanced, X_res, y_res
    gc.collect()


os.makedirs('../data/balanced', exist_ok=True)      

def stream_undersample_ddos(
    file_path,
    out_path,
    benign_per_chunk=5000,  
    chunksize=50000
):
    print("Streaming undersampling:", file_path)

    first_write = True

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        chunk.columns = chunk.columns.str.strip().str.lower()

        # keep numeric only
        chunk = chunk.select_dtypes(include=["number"])

        if "label" not in chunk.columns:
            raise ValueError("Label column missing")

        # split classes
        benign = chunk[chunk["label"] == 0]
        attack = chunk[chunk["label"] == 1]

        # sample benign
        if len(benign) > benign_per_chunk:
            benign = benign.sample(n=benign_per_chunk, random_state=42)

        # combine
        out_chunk = pd.concat([benign, attack], ignore_index=True)

        # write incrementally
        out_chunk.to_csv(
            out_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False
        )

        first_write = False

    print("Streaming undersampling completed.")



# DDOS 2019
merged_ddos = "../data/merged/ddos2019.csv"
stream_undersample_ddos(merged_ddos, '../data/balanced/ddos2019.csv')
print("Balanced DDOS Dataset generated successfully")


# IOT 2023
merged_iot = "../data/merged/iot2023.csv"
oversample(merged_iot, '../data/balanced/iot2023.csv')
print("Balanced IOT Dataset generated successfully")


# IOV 2024
merged_iov = "../data/merged/iov2024.csv"
oversample(merged_iov, '../data/balanced/iov2024.csv')
print("Balanced IOV Dataset generated successfully")

