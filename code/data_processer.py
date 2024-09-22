import kaggle
import Augmentor
import os
import shutil
import random

"""
Downloads BreakHis Dataset from Kaggle.
https://www.kaggle.com/datasets/ambarish/breakhis

- 40X, 100X, 200X, and 400X magnification
- 700x460 
- 9109 images.
"""
def get_data(data_ref):
    kaggle.api.dataset_download_files(data_ref, path="./train_data", unzip = True)
    return

"""
this function just reorganizes the data. The format obtained from kaggle does not suit my purposes.

I really do not care about type. Magnification is the only issue.

ONLY RUN THIS ONCE.
"""
def reorganize_by_mag():
    PATH = "./train_data/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/"
    DEST = "./train_data"

    #moving benign over
    #for each type
    for type in os.listdir(PATH + "benign/SOB"):
        type_path = os.path.join(PATH + "benign/SOB", type)

        #for each case
        for case in os.listdir(type_path):
            case_path = os.path.join(type_path, case)

            #for each magnifiation
            for mag in os.listdir(case_path):
                mag_path = os.path.join(case_path, mag)

                #move each image.
                for image in os.listdir(mag_path):
                    image_path = os.path.join(mag_path, image)

                    # Create destination path for the specific magnification
                    dest_path = os.path.join(DEST, mag, "benign")
                    os.makedirs(dest_path, exist_ok=True)

                    # Move the image
                    shutil.move(image_path, os.path.join(dest_path, image))
    

    for type in os.listdir(PATH + "malignant/SOB"):
        type_path = os.path.join(PATH + "malignant/SOB", type)
        for case in os.listdir(type_path):
            case_path = os.path.join(type_path, case)
            for mag in os.listdir(case_path):
                mag_path = os.path.join(case_path, mag)
                for image in os.listdir(mag_path):
                    image_path = os.path.join(mag_path, image)

                    # Create destination path for the specific magnification
                    dest_path = os.path.join(DEST, mag, "malignant")
                    os.makedirs(dest_path, exist_ok=True)

                    # Move the image
                    shutil.move(image_path, os.path.join(dest_path, image))
    
    return

"""
Split into test and train sets. Uses the 70:30 split.
"""
def train_test_split():
    PERCENT = 0.3
    PATH = "./train_data"
    DEST = "./test_data"

    #for each magnification
    for mag in os.listdir(PATH):
        mag_path = os.path.join(PATH, mag)

        #for each category
        for bm in os.listdir(mag_path):
            bm_path = os.path.join(mag_path, bm)

            #select images to move
            all_images = [i for i in os.listdir(bm_path)]
            num_to_move = int(len(all_images) * PERCENT)
            train_images = random.sample(all_images, num_to_move)

            #move images
            for image in train_images:
                source_path = os.path.join(bm_path, image)
                dest_path = os.path.join(DEST, mag, bm)

                #make path if it doesn't exist
                os.makedirs(dest_path, exist_ok=True)

                shutil.move(source_path, os.path.join(dest_path, image))
            
"""
augments the data to make each subclass have n number of samples

-Doesn't actually add new information, class inbalance still exists.
This is pretty much taken from DEEP_Pachi's method. There was no real reason to change it. 

DO NOT RUN THIS EVERY TIME. run the below method once per magnification, and only on the train data.
"""

def upsample(dir, num_samples):
    p = Augmentor.Pipeline(dir) 
    p.rotate(probability = 1, max_left_rotation = 5, max_right_rotation = 5) 
    p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.2) 
    p.skew(probability = 0.2) 
    p.shear(probability = 0.2, max_shear_left = 2, max_shear_right = 2) 
    p.crop_random(probability = 0.5, percentage_area = 0.8) 
    p.flip_random(probability = 0.2) 
    p.sample(num_samples) 
    p.random_distortion(probability = 1, grid_width = 4, grid_height = 4, magnitude = 8)
    p.flip_left_right(probability = 0.8) 
    p.flip_top_bottom(probability = 0.3) 
    p.rotate90(probability = 0.5) 
    p.rotate270(probability = 0.5)

"""
run this function ONCE to do all the upsampling for all magnifications, and ONLY ON THE TRAINING DATA
"""
def upsample_ALL(num_samples):
    PATH = "./train_data"
    for mag in os.listdir(PATH):
        dir = os.path.join(PATH, mag)
        upsample(dir, num_samples)

"""
reads and scrambles the data from the data folder

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

    # train_data = image_dataset_from_directory(DIRECTORY,
    #                                          shuffle=True,
    #                                          batch_size=BATCH_SIZE,
    #                                          image_size=IMG_SIZE,
    #                                          validation_split=SPLIT,
    #                                          subset='training',
    #                                          seed=SEED)
    # valid_data = image_dataset_from_directory(DIRECTORY,
    #                                          shuffle=True,
    #                                          batch_size=BATCH_SIZE,
    #                                          image_size=IMG_SIZE,
    #                                          validation_split=SPLIT,
    #                                          subset='validation',
    #                                          seed=SEED)

    #return stuff
    train_data = train_data.prefetch(buffer_size=data.AUTOTUNE)
    return train_data, valid_data

if __name__ == "__main__":
    #get_data("ambarish/breakhis")
    #reorganize_by_mag()
    #train_test_split()
    #upsample_ALL(1500)
    pass