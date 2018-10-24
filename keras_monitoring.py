import tensorflow as tf
from tensorflow import keras


def model_monitoring(model_path, tensorlog):
    '''
    creating the callback and the checkpoint
    - saving the current weights of the model at different points during training
    - interrupt training when validation loss has stopped improving (early stopping)
    - dynamically adjust the learning rate
    - logging train and validation metrics
    '''

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            # monitors the validation accuracy of the model
            monitor='acc',
            # stop if not improving
            patience=3,
        ),
        # can use to save the weights at each checkpoint and load the weights later on with the known model
        keras.callbacks.ModelCheckpoint(
            # path to destination model file
            filepath=model_path,
            # overwrite upon improvement of the val_loss
            monitor='val_loss',
            save_weights_only=False,
            save_best_only=True,
            verbose=1,
            mode='min',  # this is automatically inferred based on the monitor parameter
        ),
        # reduce learning rate when the validation loss has stopped improving
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            # divide the learning by 10
            factor=0.1,
            # will get triggered after the validation loss has stopped improving at least 10 epochs
            patience=2,
            min_lr=0.001,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir=tensorlog,
            # record activation histograms every 1 epoch
            histogram_freq=1,
        ),
    ]

    return callbacks_list
