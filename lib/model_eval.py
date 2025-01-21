from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt 
import re
import os
import pandas as pd
from lib import dataLoader

def train_model(model, model_name, training_loader, writer, loss_fxn, optimizer, device, epochs=25):
    """
    General function for training the model. Takes the following inputs:
    - model: a model instance to be trained
    - model_name: a string denoting what the model should be referred to when saving
    - training_loader: pytorch dataloader object referring to a valid training dataset on disk
    - writer: a SummaryWriter object for keeping track of training in real time
    - loss_fxn: a valid loss function that is compatabile with the given model
    - optimizer: an optimizer that has already been preloaded with the model parameters for training
    - device: GPU or CPU to utilize during training
    - epochs: number of training epochs. Defaults to 25.
    """

    #inidicate which model is being trained on console, and move to device
    print("Training {}:".format(model_name))
    best_loss = 1000000.
    model.to(device) 

    #epoch loop
    for e in range(1, epochs+1):
        #setting up training phase
        model.train()

        #initializing loss variables
        train_loss = 0
        train_correct = 0
        train_total = 0

        #initializing lists to hold truths and predections
        predictions = []
        truths = []

        #iterate over training data
        for i, data in enumerate(training_loader): 
            #preparing data and sending to device
            inputs, labels = data 
            inputs = inputs.to(device)
            labels = labels.to(device)

            #make predictions and calculating loss
            outputs = model(inputs) 
            loss = loss_fxn(outputs, labels) 
            train_loss += loss.item() #add loss to running loss

            #get accuracy
            _, predicted_class = torch.max(outputs.data, 1)
            batch_correct = (predicted_class == labels).sum().item()
            train_correct += batch_correct
            batch_correct /= labels.size(0)

            #add to truth/pred list
            predictions.extend(predicted_class.cpu().numpy())
            truths.extend(labels.cpu().numpy())

            #updating count
            train_total += labels.size(0)

            #logging accuracy and loss per batch
            index = e * len(training_loader) + i
            writer.add_scalar('Loss/train', loss, index)
            writer.flush()
            writer.add_scalar('Accuracy/train', batch_correct, index)
            writer.flush()

            #apply learning
            loss.backward()#get gradiant
            optimizer.step() #apply changes
            optimizer.zero_grad() #zero the gradients

        #getting overall training loss and accuracy for the epoch
        train_loss = train_loss/len(training_loader)
        train_acc, precision, recall, f1 = _log_metrics(train_correct, train_total, predictions, truths)

        #logging overall accuracy and losses
        writer.add_scalar('Training Loss',train_loss, e)
        writer.flush()
        writer.add_scalar('Training Accuracy', train_acc, e)
        writer.flush()
        writer.add_scalar('Training Precision',precision, e)
        writer.flush()
        writer.add_scalar('Training Recall', recall, e)
        writer.flush()
        writer.add_scalar('Training F1', f1, e)
        writer.flush()

        #printing results to console
        print("Epoch: {}".format(e))
        print("Training Accuracy: {}".format(train_acc))
        print("Training Precision: {}".format(precision))
        print("Training Recall: {}".format(recall))
        print("Training F1: {}".format(f1))
        print("\n")

        #saving the model if loss is better than previous:
        if train_loss < best_loss:
            best_loss = train_loss
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = 'trained_models\{}\model_{}_{}'.format(model_name, timestamp, e)
            Path('trained_models\{}'.format(model_name)).mkdir(parents=True, exist_ok=True)
            #only saving the classifier bit of the model. 
            torch.save(model.custom_classifier.state_dict(), model_path)

    writer.close()
    return
    
def validate_model(model_struct, weight_folder, eval_folder, device, last=5):
    """
    Specific function for validating muliple model weightings with multiple image magnifications. Takes the following inputs:
    - model_struct: the structure of the model to validate.
    - weight_folder: the folder path in which respective model weights are stored. 
    - eval_folder: the folder path in which evaluation data is stored. Must have subfolders in path corresponding to magnifications.
    - device: The device used compute the model output
    - last: Denotes how many of the most recent weightings to try. Defaults to 5.

    returns a dictionary of python dataframes containing the validation results, with magnification as keys. 
    """
    
    results_dict = {} 
    
    #iterate and make eval loader for every magnification in given folder.
    for mag_path in os.scandir(eval_folder):
        eval_loader = (dataLoader.make_dataloader(mag_path.path))
        mag_df = _visualize_mag_results(model_struct, weight_folder, eval_loader, device, last)
        #add to results dictionary
        results_dict[os.path.basename(mag_path.path)] = mag_df
    return results_dict


def _visualize_mag_results(model_struct, folder_path, eval_loader, device, last):
    """
    Hidden Function for making and saving accuracy, precision, recall, and f1 comparison graphs.
    """
    #dataframe for storing acc, prec, recall, and f1 score
    df = pd.DataFrame(columns = ["Accuracy", "Precision", "Recall", "f1"])

    #taking only the last few models in folder to prevent optimization bias
    for weights in os.listdir(folder_path)[-last:]:
        num = re.search(r"(?<=_)\d+$", weights).group()
        df.loc[num] = _eval_model(model_struct, os.path.join(folder_path, weights), eval_loader, device)

    #subfunction for adding data labels to graphs
    def add_labels(axs, df_series):
        for tup in df_series.items():
            axs.text(tup[0], tup[1], f'{tup[1]:.2f}', ha='center')

    #plotting results
    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axs[0].title.set_text('Accuracy')
    axs[0].bar(df.index, df["Accuracy"], color = 'tab:red')
    add_labels(axs[0], df["Accuracy"])
    axs[1].title.set_text('Precision')
    axs[1].bar(df.index, df["Precision"], color = 'tab:orange')
    add_labels(axs[1], df["Precision"])
    axs[2].title.set_text('Recall')
    axs[2].bar(df.index, df["Recall"], color = 'tab:green')
    add_labels(axs[2], df["Recall"])
    axs[3].title.set_text('F1 Score')
    axs[3].bar(df.index, df["f1"], color = 'tab:blue')
    add_labels(axs[3], df["f1"])
    
    valid_mag = os.path.basename(eval_loader.dataset.root)
    model_name = os.path.basename(os.path.normpath(folder_path))
    title = fig.suptitle("Validation Metrics for {} at {}".format(model_name, valid_mag), y = 1.07)
    xlabel = fig.supxlabel("Model Number", y = -0.07)

    #saving graph
    save_at = os.path.join("validation_graphs", model_name, "{}_{}".format(model_name, valid_mag))
    plt.savefig(save_at, bbox_inches='tight', bbox_extra_artists=[title, xlabel])
    plt.close(fig)

    #return results dataframe for further computation
    return df

    
def _eval_model(model_struct, weights_path, eval_loader, device):
    """
    Hidden function for getting actuall accuracy, precision, recall, and f1 scores for each model. 
    """
    model = model_struct
    
    # Model class must be defined somewhere
    model.custom_classifier.load_state_dict(torch.load(weights_path, weights_only=True))
    model.to(device)
    model.eval()

    #initializing lists to hold truths and predections
    eval_correct = 0
    eval_total = 0
    predictions = []
    truths = []

    for i, data in enumerate(eval_loader): 
        #preparing data and sending to device
        inputs, labels = data 
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        #make predictions
        outputs = model(inputs) 
        _, predicted_class = torch.max(outputs.data, 1)

        #add to truth/pred list
        predictions.extend(predicted_class.cpu().numpy())
        truths.extend(labels.cpu().numpy())

        #adding to total correct predictions
        eval_correct += (predicted_class == labels).sum().item()
        eval_total += labels.size(0)

    accuracy, precision, recall, f1 = _log_metrics(eval_correct, eval_total, predictions, truths)
    return [accuracy, precision, recall, f1]


def _log_metrics(correct, total, predictions, truths):
    """
    Hidden function to calculate accuracy, precision, recall, and f1"
    """
    
    acc = correct/total
    #getting precision, recall, and f1 score
    precision = precision_score(truths, predictions)
    recall = recall_score(truths, predictions)
    f1 = f1_score(truths, predictions)
    return acc, precision, recall, f1