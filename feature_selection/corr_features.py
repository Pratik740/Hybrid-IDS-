import pandas as pd
import numpy as np

reader = pd.read_csv(
    "../data/balanced/ddos2019.csv",
    chunksize=200_000,
    on_bad_lines="skip"
)

corr_sum = None
num_chunks = 0


for chunk in reader:
    if "label" not in chunk.columns:
        continue

    # Feature-feature correlation only
    X = chunk.drop(columns=["label"])

    # Compute correlation for this chunk
    corr = X.corr()

    if corr_sum is None:
        corr_sum = corr
    else:
        corr_sum += corr

    num_chunks += 1
    print(f"Processed chunk {num_chunks}")

corr_matrix = (corr_sum / num_chunks).abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [
    col for col in upper_triangle.columns
    if any(upper_triangle[col] > 0.9)
]

print(f"Total chunks processed: {num_chunks}")
print(f"Features before filtering: {corr_matrix.shape[0]}")
print(f"Features removed (correlated): {len(to_drop)}")
print(f"Features remaining: {corr_matrix.shape[0] - len(to_drop)}")

pd.Series(to_drop).to_csv("correlation_dropped_features.txt", index=False)


# Load dropped features
dropped = pd.read_csv("correlation_dropped_features.txt", header=None)[0].tolist()

# Load feature names ONLY (header read, not full file)
all_features = pd.read_csv(
    "../data/balanced/ddos2019.csv",
    nrows=0
).columns.tolist()

all_features.remove("label")

remaining_features = [f for f in all_features if f not in dropped]

pd.Series(remaining_features).to_csv(
    "./correlation_selected_features.txt",
    index=False
)

print("Saved remaining features:", len(remaining_features))

