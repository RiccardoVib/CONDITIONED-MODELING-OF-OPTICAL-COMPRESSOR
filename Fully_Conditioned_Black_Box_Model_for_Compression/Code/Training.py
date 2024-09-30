# Copyright (C) 2023 Riccardo Simionato, University of Oslo
# Inquiries: riccardo.simionato.vib@gmail.com.com
#
# This code is free software: you can redistribute it and/or modify it under the terms
# of the GNU Lesser General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
#
# This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Less General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this code.
# If not, see <http://www.gnu.org/licenses/>.
#
# If you use this code or any part of it in any program or publication, please acknowledge
# its authors by adding a reference to this publication:
#
# R. Simionato, 2023, "Fully Conditioned Black Box Model for Compression" in proceedings of the 23th Digital Audio Effect Conference, Copenaghen, Denmark.



import os
import tensorflow as tf
from UtilsForTrainings import plotTraining, writeResults, checkpoints, predictWaves
from Models import create_model_ED_CNN
from DatasetsClass import DataGeneratorCL1B, DataGeneratorLA2A
import numpy as np
import random

def train(data_dir, epochs, seed=422, **kwargs):
    """
      :param data_dir: the directory in which dataset are stored [string]
      :param b_size: the size of each batch [int]
      :param learning_rate: the initial leanring rate [float]
      :param units: the number of model's units [int]
      :param cond: the number of conditioning parameters [int]
      :param model_save_dir: the directory in which models are stored [string]
      :param save_folder: the directory in which the model will be saved [string]
      :param inference: if True it skip the training and it compute only the inference [bool]
      :param w_length: input size [int]
      :param activation: activation function [string]
      :param out: the output size [int]

    """
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.0001)
    units = kwargs.get('units', 64)
    D = kwargs.get('cond', 4)
    model_save_dir = kwargs.get('model_save_dir', '/scratch/users/riccarsi/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    T = kwargs.get('w_length', 16)
    inference = kwargs.get('inference', False)
    activation = kwargs.get('activation', 'sigmoid')
    o = kwargs.get('out', 16)

    if D == 4:
        data_generator = DataGeneratorCL1B
        filename = 'TubeTech'
        fs = 48000
    else:
        data_generator = DataGeneratorLA2A
        filename = 'LA2A'
        fs = 44100

    # set all the seed in case reproducibility is desired
    #np.random.seed(422)
    #tf.random.set_seed(422)
    #random.seed(422)


    # check if GPUs are available and set the memory growing
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if len(gpu) != 0:
        tf.config.experimental.set_memory_growth(gpu[0], True)
        
  
    # create the model
    model = create_model_ED_CNN(D, T, units, activation)
    # define the Adam optimizer with the initial learning rate
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1)
    # compile the model
    model.compile(loss='mse', metrics=['mse', 'mae'], optimizer=opt)

    # define callbacks: where to store the weights
    callbacks = []
   
    ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest = checkpoints(model_save_dir, save_folder)
    callbacks += [ckpt_callback, ckpt_callback_latest]

    if not inference:
        # load the weights of the model
        last = tf.train.latest_checkpoint(ckpt_dir_latest)
        if last is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(last).expect_partial()
        else:
            # if no weights are found, they are initialized
            print("Initializing random weights.")

        #train_data
        train_gen = data_generator(filename + "_train.pickle", data_dir, input_enc_size=T//2, input_dec_size=T//2, output_size=o, cond_size=D, shuffle=False, batch_size=b_size)
        #val_data
        val_gen = data_generator(filename + "_test.pickle", data_dir, input_enc_size=T//2, input_dec_size=T//2, output_size=o, cond_size=D, shuffle=False, batch_size=b_size)

        results = model.fit(train_gen, epochs=epochs, verbose=0, validation_data=val_gen, shuffle=False, callbacks=callbacks)

        writeResults(None, results, b_size, learning_rate, model_save_dir, save_folder, 0)

        loss_training = results.history['loss']
        loss_val = results.history['val_loss']

        plotTraining(loss_training, loss_val, model_save_dir, save_folder)

        print("Training done")
    
    # load the best weights of the model
    best = tf.train.latest_checkpoint(ckpt_dir)
    if best is not None:
        print("Restored weights from {}".format(ckpt_dir))
        model.load_weights(best).expect_partial()
    else:
        # if no weights are found, there is something wrong
        print("Something is wrong.")
        
    test_gen = data_generator(filename + "_test.pickle", data_dir, input_enc_size=T//2, input_dec_size=T//2, output_size=o, cond_size=D, shuffle=False, batch_size=b_size)

    # compute test loss
    test_loss = model.evaluate(test_gen, verbose=0, return_dict=True)

    print('Test Loss: ', test_loss)
    predictions = model.predict(test_gen, verbose=0)
    # plot and render the output audio file, together with the input and target
    predictWaves(predictions, test_gen.x.reshape(-1), test_gen.y.reshape(-1), model_save_dir, save_folder, fs)

    results_ = {'Test_Loss': test_loss}

    # write and store the metrics values
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
        for key, value in results_.items():
            print('\n', key, '  : ', value, file=f)

    return results_


if __name__ == '__main__':
    data_dir = '../Files' # data folder to dataset
    #data_dir = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/Compressors/DatasetOLD' # data folder to dataset
    seed = 422 # seed in case reproducibility is desired

    train(data_dir=data_dir,
            model_save_dir='../Models/',
            save_folder='CL 1B Model',
            b_size=1,
            learning_rate=0.0001,
            units=64,
            epochs=1,
            activation='sigmoid',
            w_length=32,
            out=16,
            cond=4,
            inference=False)
