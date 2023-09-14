import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from librosa import display
from scipy.io import wavfile


# class VASchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, initial_learning_rate):
#         super(VASchedule, self).__init__()
#         self.initial_learning_rate = initial_learning_rate
#         #self.decay_steps = decay_steps
#         #self.decay_rate = decay_rate
#         #self.min_learning_rate = min_learning_rate
#
#     def __call__(self, step):
#         initial_learning_rate = tf.cast(self.initial_learning_rate, tf.float32)
#         #decay_steps = tf.cast(self.decay_steps, tf.float32)
#         #decay_rate = tf.cast(self.decay_rate, tf.float32)
#         #global_step = tf.cast(step, tf.float32)
#
#         learning_rate = initial_learning_rate / (1 + decay_rate * (global_step / decay_steps))
#         return tf.maximum(learning_rate, self.min_learning_rate)
#
#     def get_config(self):
#         return {
#             "initial_learning_rate": self.initial_learning_rate,
#             "decay_steps": self.decay_steps,
#             "decay_rate": self.decay_rate,
#             "min_learning_rate": self.min_learning_rate,
#         }

class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, min_learning_rate, decay_rate):
        super(CustomLRSchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.decay_rate = decay_rate

    def __call__(self, step):
        return tf.maximum(self.initial_learning_rate * tf.math.pow(self.decay_rate, step), self.min_learning_rate)

    def get_config(self):
        config = {
            "initial_learning_rate": self.initial_learning_rate,
            "min_learning_rate": self.min_learning_rate,
            "decay_rate": self.decay_rate,
        }
        return config
    
def writeResults(test_loss, results, b_size, learning_rate, model_save_dir, save_folder,
                 index):
    results = {
        'Test_Loss': test_loss,
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        # 'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss']
    }
    with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.txt'])), 'w') as f:
        for key, value in results.items():
            print('\n', key, '  : ', value, file=f)
        pickle.dump(results,
                    open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_' + str(index) + '.pkl'])),
                         'wb'))


def plotResult(predictions, y, model_save_dir, save_folder):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # ax.plot(predictions, label='pred')
    # ax.plot(x, label='inp')
    # ax.plot(y, label='tar')
    display.waveshow(y, sr=48000, ax=ax, label='Target', alpha=0.9)
    display.waveshow(predictions, sr=48000, ax=ax, label='Prediction', alpha=0.7)
    # ax.label_outer()
    ax.legend(loc='upper right')
    fig.savefig(model_save_dir + '/' + save_folder + '/plot')
    plt.close('all')


def plotTraining(loss_training, loss_val, model_save_dir, save_folder):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(np.array(loss_training), label='train'),
    ax.plot(np.array(loss_val), label='validation')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.title('train vs. validation accuracy')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
    fig.savefig(model_save_dir + '/' + save_folder + '/' + 'loss.png')
    plt.close('all')


def predictWaves(predictions, x_test, y_test, model_save_dir, save_folder, fs):
    pred_name = '_pred.wav'
    inp_name = '_inp.wav'
    tar_name = '_tar.wav'

    pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
    tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

    if not os.path.exists(os.path.dirname(pred_dir)):
        os.makedirs(os.path.dirname(pred_dir))

    wavfile.write(pred_dir, fs, predictions.reshape(-1))
    wavfile.write(inp_dir, fs, x_test.reshape(-1))
    wavfile.write(tar_dir, fs, y_test.reshape(-1))

    plotResult(predictions.reshape(-1), y_test.reshape(-1), model_save_dir, save_folder)


def checkpoints(model_save_dir, save_folder):
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

    return ckpt_callback, ckpt_callback_latest, ckpt_dir, ckpt_dir_latest