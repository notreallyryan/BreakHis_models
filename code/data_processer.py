from keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from tensorflow import data

"""
reads and scrambles the data from the BreaKHis_sorted folder

inputs: magnificaton - specificaton of which magnification dataset to use

outputs: an ndarray of images, and an ndarray of labels, both shuffled accordingly
"""
def read_data(magnification = '100x'):

    #variable stuff
    BATCH_SIZE = 32
    IMG_SIZE = (460, 700)
    SPLIT = 0.2
    DIRECTORY = "../BreaKHis_training/" + magnification + "/"
    SEED = 42 #"The meaning of life, the universe, and everything"

    train_data = image_dataset_from_directory(DIRECTORY,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=SPLIT,
                                             subset='training',
                                             seed=SEED)
    valid_data = image_dataset_from_directory(DIRECTORY,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=SPLIT,
                                             subset='validation',
                                             seed=SEED)

    #return stuff
    train_data = train_data.prefetch(buffer_size=data.AUTOTUNE)
    return train_data, valid_data


if __name__ == "__main__":
    read_data('small') 
    pass