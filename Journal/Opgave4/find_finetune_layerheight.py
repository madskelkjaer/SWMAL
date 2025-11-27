import tensorflow as tf

# 1) Load the baseline model with trained head
model = tf.keras.models.load_model("cats_feature_extractor.keras")

# 2) Get handle to the base_model inside it (depends on how you built it)
base_model = model.get_layer("inception_v3")  # or whatever name you used

def run_fine_tune(fine_tune_from, lr=1e-5):
    # unfreeze from chosen layer
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=[early_stop]  # same EarlyStopping as before
    )
    return max(history_ft.history["val_accuracy"])
