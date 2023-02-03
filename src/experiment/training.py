import tensorflow.keras as keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

from src.experiment.hyperparameter import batch_size, num_epoch_finetune, learning_rate
from src.infra.model import create_sp_classifier


def train_classifier(log_dir, x_train, x_train_5min, y_train, x_val, x_val_5min, y_val):
    # y_train = keras.utils.to_categorical(y_train, num_classes=2)  # Convert to two categories
    # y_val = keras.utils.to_categorical(y_val, num_classes=2)  # Convert to two categories

    classifier = create_sp_classifier()

    classifier.compile(loss=keras.losses.CategoricalCrossentropy(), optimizer=keras.optimizers.Adam(learning_rate),
                       metrics=[keras.metrics.CategoricalAccuracy()])
    classifier.summary()
    filepath = log_dir + '/checkpoint/classifier.weights.best.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                 mode='min')

    redonplat = ReduceLROnPlateau(monitor="val_loss", mode="min", patience=5, verbose=2)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=15)
    callbacks_list = [redonplat, checkpoint, es]
    history = classifier.fit(x=[x_train, x_train_5min], y=y_train, validation_data=([x_val, x_val_5min], y_val),
                             batch_size=batch_size,
                             epochs=num_epoch_finetune,
                             callbacks=callbacks_list)
