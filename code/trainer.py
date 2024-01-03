from data_processer import read_data
from pandas import DataFrame
from os.path import exists
from os import makedirs
import matplotlib.pyplot as plt
from shutil import rmtree

"""
Trains a given compiled model, produces the accuracy and learning loss graphs 
"""

def train_model(imgsize, untrained_model, magnification, epochs):
    MODELNAME = str(untrained_model.__name__) + "_" + str(magnification)
    #obtain specified model, training and validation data
    train_data, valid_data = read_data(magnification)
    model = untrained_model(imgsize)

    #fitting the model
    history = model.fit(train_data, epochs = epochs, validation_data=valid_data, batch_size=32, verbose=2)

    #creating trained_model folder
    FOLDER = "./trained_models/" + MODELNAME
    if exists(FOLDER):
        rmtree(FOLDER)

    makedirs(FOLDER)

    #plot accuracies and learning loss + save them
    save_graphs(history, FOLDER)

    #save metric scores
    save_scores(history, FOLDER)
    
    #saves the model. This was written as a seperate function cuz i might want to add some comaprison/overwrite features later
    save_model(model, FOLDER)

    return


"""
Saves the trained model for future use. 
"""
def save_model(trained_model, folder_name):
    FILE_NAME = folder_name + "/model.keras"
        
    trained_model.save(FILE_NAME)
    print("Model saved successfully.")
    
    return


"""
saves metric scores as a txt file for future reference
"""
def save_scores(history, folder_name):
    FILE_NAME = folder_name + "/scores.txt"

    readfrom = history.history
    rows = ["training", "validation"]
    cols = ["loss", "accuracy", "precision", "recall", "f1"]

    df = DataFrame(columns = cols)
    
    for metric in cols:
        temp = []
        temp.append(readfrom.get(metric)[-1])
        temp.append(readfrom.get("val_" + metric)[-1])
        df[metric] = temp

    df.index = rows
    
    with open(FILE_NAME, 'a') as f:
        tosave = df.to_string()
        f.write(tosave)

    return


"""
save graphics in for future reference
"""
def save_graphs(history, folder_name):
    loss_acc_df = DataFrame(history.history)

    losses = loss_acc_df[['loss','val_loss']]
    losses.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
    losses.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
    plt.savefig(folder_name + "/losses.png")

    accuracies= loss_acc_df[['accuracy','val_accuracy']]
    accuracies.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
    accuracies.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
    plt.savefig(folder_name + "/accuracy.png")

    precision= loss_acc_df[['precision','val_precision']]
    precision.rename(columns={'precision':'train','val_precision':'validation'},inplace=True)
    precision.plot(title='Model precision',figsize=(12,8)).set(xlabel='Epoch',ylabel='Precision')
    plt.savefig(folder_name + "/precision.png")

    recall= loss_acc_df[['recall','val_recall']]
    recall.rename(columns={'recall':'train','val_recall':'validation'},inplace=True)
    recall.plot(title='Model recall',figsize=(12,8)).set(xlabel='Epoch',ylabel='Recall')
    plt.savefig(folder_name + "/recall.png")

    f1= loss_acc_df[['f1','val_f1']]
    f1.rename(columns={'f1':'train','val_f1':'validation'},inplace=True)
    f1.plot(title='Model f1 Score',figsize=(12,8)).set(xlabel='Epoch',ylabel='f1 Score')
    plt.savefig(folder_name + "/f1.png")


    
#testing stuff
if __name__ == "__main__":
    pass