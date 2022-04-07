import numpy as np
import os
import tensorflow as tf
from GetData import get_data
from scipy.io import wavfile
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pickle


def trainDense(data_dir, epochs, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 128)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('units', [32, 32])
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'FFN')
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 1 )
    act = kwargs.get('act', 'tanh')
    scaler = None

    if inference:
        file_scaler = open(os.path.normpath('/'.join([data_dir, 'scaler.pickle'])), 'rb')
        scaler = pickle.load(file_scaler)

    x, y, x_val, y_val, x_test, y_test, scaler, fs = get_data(data_dir=data_dir, w_length=w_length, inference=inference, scaler=scaler,
                                                                              seed=seed)

    layers = len(units)
    n_units = ''
    for unit in units:
        n_units += str(unit) + ', '

    n_units = n_units[:-2]

    # T past values used to predict the next value
    T = x.shape[1]  # time window
    D = x.shape[2]  # features

    inputs = Input(shape=(T, D), name='input')
    first_unit = units.pop(0)
    if len(units) > 0:
        last_unit = units.pop()
        outputs = Dense(first_unit, name='Dense_0')(inputs)
        for i, unit in enumerate(units):
            outputs = Dense(unit, name='Dense_' + str(i + 1))(outputs)
        outputs = Dense(last_unit, activation=act, name='Dense_Fin')(outputs)
    else:
        outputs = Dense(first_unit, activation=act, name='Dense')(inputs)

    final_outputs = Dense(1, name='DenseLay')(outputs)
    model = Model(inputs, final_outputs)
    model.summary()

    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    if loss_type == 'mae':
        model.compile(loss='mae', metrics=['mae'], optimizer=opt)
    elif loss_type == 'mse':
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')

    callbacks = []
    if ckpt_flag:
        ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
        ckpt_path_latest = os.path.normpath(
            os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
        ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
        ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
            os.makedirs(os.path.dirname(ckpt_dir_latest))

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                           save_best_only=True, save_weights_only=True, verbose=1)
        ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                                  mode='min',
                                                                  save_best_only=False, save_weights_only=True,
                                                                  verbose=1)
        callbacks += [ckpt_callback, ckpt_callback_latest]
        latest = tf.train.latest_checkpoint(ckpt_dir_latest)
        if latest is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(latest)
            # start_epoch = int(latest.split('-')[-1].split('.')[0])
            # print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.000001, patience=20,
                                                               restore_best_weights=True,
                                                               verbose=0)
    callbacks += [early_stopping_callback]
    # train
    if not inference:
        results = model.fit(x, y, batch_size=b_size, epochs=epochs,
                            validation_data=(x_val, y_val), callbacks=callbacks, verbose=0)

    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_loss = model.evaluate(x_test, y_test, batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    if inference:
        results = {}
    else:
        results = {
            'Test_Loss': test_loss,
            'Min_val_loss': np.min(results.history['val_loss']),
            'Min_train_loss': np.min(results.history['loss']),
            'b_size': b_size,
            'learning_rate': learning_rate,
            'opt_type': opt_type,
            'loss_type': loss_type,
            'layers': layers,
            'n_units': n_units,
            'w_length': w_length,
            # 'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss']
        }
        print(results)
    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    if generate_wav is not None:

        predictions = model.predict(x_test)

        print('GenerateWavLoss: ', model.evaluate(x_test, y_test, batch_size=b_size, verbose=0))
        predictions = scaler[0].inverse_transform(predictions)
        x_test = scaler[0].inverse_transform(x_test[:, :, 0])
        y_gen = scaler[0].inverse_transform(y_test)

        predictions = predictions.reshape(-1)
        x_test = x_test.reshape(-1)
        y_test = y_test.reshape(-1)

        # Define directories
        pred_name = 'FFN_pred.wav'
        inp_name = 'FFN_inp.wav'
        tar_name = 'FFN_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        # Save Wav files
        predictions = predictions.astype('int16')
        x_test = x_test.astype('int16')
        y_test = y_test.astype('int16')
        wavfile.write(pred_dir, int(fs), predictions)
        wavfile.write(inp_dir, int(fs), x_test)
        wavfile.write(tar_dir, int(fs), y_test)

    return results


if __name__ == '__main__':
    data_dir = './Files'

    seed = 422
    trainDense(data_dir=data_dir,
               model_save_dir='../../TrainedModels',
               save_folder='FFN',
               ckpt_flag=True,
               b_size=128,
               learning_rate=0.001,
               units=[32, 32],
               epochs=100,
               loss_type='mse',
               w_length=1,
               act='tanh',
               inference=False)