from keras import models
import trainer
import os
import executor
import model_structures

"""
to be called by the user.
"""

"""
trains a selected model on the specified BreakHis Data
"""
def train_select_model(model_name, magnification, epochs):
    IMGSIZE = (460, 700, 3)
    EPOCHS = epochs

    #check if model acutally exists in models.py file
    if model_name not in dir(model_structures):
        print("no such model found")
        return
    
    else:
        untrained_model = getattr(model_structures, model_name)

    trainer.train_model(IMGSIZE, untrained_model, magnification, EPOCHS)
    print("Model trained successfully!")
    return
    


"""
applies a trained model on specified data in "New_data"
"""
def use_select_model(data_folder, model_name, magnification):
    FILENAME = "./trained_models/" + model_name + "_" + magnification + ".keras"

    if os.path.exists(FILENAME):
        print("file found")
        loaded_model = model_structures.load_model(FILENAME)
        executor.apply_model(data_folder, loaded_model)
    
    else:
        print("file not found")



#for testing purposes
if __name__ == "__main__":
    #train_select_model("MobileNetV2_base", 'small', 50)
    #train_select_model("ResNet50v2_base", 'small', 50)
    #this has some issue saving the model
    #train_select_model("VGG19_base", 'small', 50) 
    train_select_model("custom_model", 'small', 25) 
    pass