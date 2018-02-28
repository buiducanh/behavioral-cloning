from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, BatchNormalization, Activation, Cropping2D
from data import DataPipeline
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_root', 'data/sampledata/', "root of the data set")
flags.DEFINE_string('model_name', 'model.h5', "root of the data set")
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('seed', 42, "The number of epochs.")
flags.DEFINE_integer('batch_size', 128, "The batch size.")

def network():
    # Nvidia Model
    model = Sequential()

    model.add(Cropping2D(cropping=((60,30), (0,0)), input_shape=(160, 160, 3)))

    model.add(Lambda(lambda x: (x / 127.5) - 1))

    # First Convolutional Layer
    model.add(Conv2D(nb_filter = 24, nb_row = 5, nb_col = 5, subsample = (2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    # Second Convolutional Layer
    model.add(Conv2D(nb_filter = 36, nb_row = 5, nb_col = 5, subsample = (2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    # Third Convolutional Layer
    model.add(Conv2D(nb_filter = 48, nb_row = 5, nb_col = 5, subsample = (2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    # Fourth Convolutional Layer
    model.add(Conv2D(nb_filter = 64, nb_row = 3, nb_col = 3, subsample = (1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    # Fifth Convolutional Layer
    model.add(Conv2D(nb_filter = 64, nb_row = 3, nb_col = 3, subsample = (1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    model.add(Flatten())

    # First Fully Connected Layer
    model.add(Dense(100))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    # Second Fully Connected Layer
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    # Third Fully Connected Layer
    model.add(Dense(10))
    model.add(BatchNormalization())
    model.add(Activation('elu'))

    # Output
    model.add(Dense(1))

    # Choose optimizer and loss
    model.compile(loss = 'mse', optimizer= 'adam')

    return model

def main():
    np.random.seed(FLAGS.seed)

    model_name = FLAGS.model_name

    try:
        open(model_name)
        print('Model already learnt')
    except OSError:
        model = network()
        data = DataPipeline(data_rootseed = FLAGS.seed)
        num_train, num_valid = data.num_training(), data.num_validation()

        model.fit_generator(
                data.generate(batch_size = FLAGS.batch_size),
                samples_per_epoch = num_train ,
                validation_data = data.generate(batch_size = BATCH_SIZE, validation = True),
                nb_val_samples = num_valid,
                nb_epoch = FLAGS.epochs)

        model.save(model_name)

if __name__ == '__main__':
    tf.app.run()
