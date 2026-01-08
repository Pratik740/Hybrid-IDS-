from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization 
from tensorflow.keras.optimizers import Adam

def build_cascaded_lstm(n_features):
    model = Sequential()

    model.add(Input(shape=(1, n_features)))

    # LSTM Layer 1
    print("Training LSTM Layer 1 ...")
    model.add(LSTM(50, return_sequences=True)    )
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # LSTM Layer 2
    print("Training LSTM Layer 2 ...")
    model.add(LSTM(50, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # LSTM Layer 3
    print("Training LSTM Layer 3 ...")
    model.add(LSTM(50, return_sequences=False))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # We won't use dense layer and just return embeddings (feature-representation vector of size = 50) received after dropout layer.
    model.add(Dense(1, activation="sigmoid"))  

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model



