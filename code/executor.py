from keras.utils import image_dataset_from_directory

"""
Use model on new data
"""

def apply_model(data_folder, loaded_model):
    DIRECTORY = "./New_data/" + data_folder + "/"

    new_data = image_dataset_from_directory(
        directory = DIRECTORY,
        labels = None,
        shuffle = False
    )

    predictions = loaded_model.predict(new_data)

    print(predictions)

    return

#for testing purposes
if __name__ == "__main__":
    apply_model("test")