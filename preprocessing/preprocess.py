import pandas as pd
import numpy as np
from pathlib import Path

import glob
import os


def merge_and_clean(files, attack_keywords, output_path, chunksize=50_000):
    os.makedirs(Path(output_path).parent, exist_ok=True)

    pattern = "|".join(["benign"] + attack_keywords)
    first_write = True

    for f in files:
        print(f"Processing {f}")

        for df in pd.read_csv(f, chunksize=chunksize):   # df is chunk of file f
            df.columns = df.columns.str.strip().str.lower()

            if 'label' not in df.columns:
                print(f"Skipping {f}: label column missing")
                break

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=['label'], inplace=True)

            df = df[df['label'].str.contains(pattern, case=False, na=False)]
            if df.empty:
                continue

            df['label'] = (
                df['label']
                .str.lower()
                .ne('benign')
                .astype('int8')
            )

            df.to_csv(
                output_path,
                mode='w' if first_write else 'a',
                header=first_write,
                index=False
            )
            print("chunk processed", flush=True)
            first_write = False


os.makedirs('../data/merged', exist_ok=True)      


# DDOS 2019
ddos_files = glob.glob("../data/raw/ddos2019/*.csv")
merge_and_clean(ddos_files,
     ['NTP','UDP','SYN'],
     '../data/merged/ddos2019.csv')
print("Merged Successfully ddos")


# IOT 2023
iot_files = glob.glob("../data/raw/iot2023/*.csv")
merge_and_clean(iot_files,
     ['SPOOFING'],
     '../data/merged/iot2023.csv')
print("Merged Successfully iot")


# IOV 2024
iov_files = glob.glob("../data/raw/iov2024/*.csv")
merge_and_clean(iov_files,
    ['SPOOFING','ATTACK'],
    '../data/merged/iov2024.csv')
print("Merged Successfully iov")




