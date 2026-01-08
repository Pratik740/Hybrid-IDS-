import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def train_hybrid(df, model_builder, save_prefix):

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = model_builder["prepare"](df)

    # Build LSTM
    n_features = X_train.shape[2]
    lstm = model_builder["model"](n_features)

    # Train LSTM
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    # It is done so that if validation score doesn't improve for 5 consecutive epoch, it will stop and 
    # rollbacks to bestfound weights

    lstm.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    # Remove Dense layer → feature extractor
    feature_extractor = Model(
        inputs=lstm.inputs,
        outputs=lstm.layers[-2].output    # Coz last layer is dense layer(for classification using sigmoid) but we classiying using naive-bayes.
    )

    X_train_feat = feature_extractor.predict(X_train)
    X_test_feat = feature_extractor.predict(X_test)

    # Train Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_feat, y_train)

    y_pred = nb.predict(X_test_feat)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test Accuracy ({save_prefix}): {acc:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artifacts
    lstm.save(f"../models/{save_prefix}_lstm.h5")
    joblib.dump(nb, f"../models/{save_prefix}_nb.pkl")
    joblib.dump(scaler, f"../models/{save_prefix}_scaler.pkl")

    print(f"✅ Finished training for {save_prefix}")
