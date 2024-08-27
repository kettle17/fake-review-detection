#Imports
import random
from tqdm.auto import tqdm
from transformers import RobertaTokenizer
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
from transformers import RobertaTokenizer, RobertaModel
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import os
from decimal import *
import numbers
import copy
import sys
import gc

#importing other notebooks, CHANGE in py
from general_functions import * #load functions
from class_define import * #load classes

tqdm.pandas()

#CONSTANT CONSTANTS
NUM_EPOCHS = 100 #number of epochs
MAX_LENGTH = 100 #maximum length of each review from dataset. if the review is longer then it will be cut at this point
BATCH_SIZE = 8 #batch size of each run
SEED = 42 #Seed for reproducibility

#These will be changed by the program later
LEARNING_RATE = 0.0001 #learning rate for optimizer
STEP_SIZE = 12 #number of epochs before learning rate decreased
GAMMA = 0.25 #amount learning rate is decreased by each step size (lr * gamma)
WEIGHT_DECAY=0.002 #weight decay to prevent overfit for optimizer

# setting seed defined in constants
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# Additional settings for PyTorch
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for multi-GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #The device used for model allocation. If the GPU is available, it will be used, otherwise the CPU will be used.

print(device)
if (str(device) != "cuda:0"): #The CPU takes exponentially longer to train models, so a warning is output if the GPU cannot be allocated.
	print("WARNING: CUDA NOT FOUND. DEVICE IS CPU")

#=================User Input Section 1: Dataset loading==================
#Constitutes user choice of dataset: FRDDS, or custom
chk = ""
mode = 0
datasetmode = 0
dataset = ""

#The program will prompt the user for their choice of dataset...
while chk == "":
    print("Would you like to use the FRDDS dataset, or to use a customdataset?")
    print("1. FRDDS dataset")
    print("2. Custom dataset")
    chk = str(input("Please enter 1 or 2:")).lower() #Prompt for dataset
    if (chk == "1" or chk == "frdds" or chk == "frdds dataset"):
        print("FRDDS")
        mode=1 #If so, skip to after path entry
    elif (chk == "2" or chk == "custom" or chk == "custom dataset"):
        print("Custom")
        mode=2 #If chosenm proceed to path entry
    else:
        print("Not valid input") #Incorrect input
        chk = ""

#Program will take a name of csv present in the customdataset folder
if (mode == 2):
     chk = ""
     while chk == "":
        chk = str(input("Please enter the name of the custom dataset:")).lower() #Prompt for custom dataset name
        try:
            dataset = dataset_load("dataset/custom/" + chk + ".csv", ["item_title", "rating", "timestamp", "images"], 'utf8') #call custom dataset
        except:
            print("File does not exist/Error")
            chk = ""
else: #else load FRDDS
    try:
        #Calling dataset load function. Strips dataset of unneeded columns as in feature analysis (saves loading).
        dataset = dataset_load("dataset/shuffleFRDDS.csv", ["item_title", "rating", "timestamp", "images"], 'utf8')
        print("FRDDS loaded.")
    except:
        print("shuffleFRDDS is not present.")
        raise SystemExit(0)

#Label instruction for label conversion
originallabels = {
    'cg' : 0,
    'or' : 1
} 

#Calling label convert function to convert labels in dataset to numerals
dataset_label_convert(dataset, originallabels)

#The program will now prompt user for train:eval:test splits.
chk =""
test_size = 0.175
eval_size = 0.21212121212
while chk == "":
    print("Would you like to use the preset train:eval:test splits, or to use a custom split?")
    print("1. Preset splits")
    print("2. Custom splits")
    chk = str(input("Please enter 1 or 2:")).lower() #Prompt for dataset
    if (chk == "1" or chk == "preset" or chk == "preset splits"):
        print("Presets")
        mode=1 #If so, skip to after path entry
    elif (chk == "2" or chk == "custom" or chk == "custom splits"):
        print("Custom")
        mode=2 #If chosen proceed to path entry
    else:
        print("Not valid input") #Incorrect input
        chk = ""
#Program will take a name of csv present in the customdataset folder
if (mode == 2):
    chk = ""
    while chk == "":
        test_size = 0
        eval_size = 0
        print("Please enter your splits:")
        chk = int(input("1. Training:")) / 100 #Prompt for training
        eval_size = int(input("2. Evaluation:")) / 100 #Prompt for eval
        print(chk, eval_size)
        test_size = round((1 - chk - eval_size), 8)
        print("Please confirm these are the correct splits:")
        print("Training: " + str(chk*100) + "%, Evaluation: " + str(eval_size*100) + "%, Testing: " + str(test_size*100) + "%")
        chk = str(input("Please confirm yes or no:")).lower() #Prompt for dataset
        if (chk == "yes" or chk == "y" or chk == ""):
            print("Yes")
            chk = "chk"
        else:
            print("No")
            chk = ""

chk = ""
#The program will then prompt the user for their choice in dataset mode:
while chk == "":
    print("What dataset mode would you like to use?")
    print("1. Review text only (default)")
    print("2. Review title and text")
    print("3. Review title only")
    chk = str(input("Please enter 1, 2 or 3:")).lower() #Prompt for dataset
    if (chk == "1" or chk == "text" or chk == "text only" or chk == "review text" or chk == "review text only"):
        print("Review text only")
        datasetmode=1 
    elif (chk == "2" or chk == "title and text" or chk == "titletext" or chk == "review title and text"):
        print("Review title and text")
        datasetmode=2
    elif (chk == "3" or chk == "title" or chk == "title only" or chk == "review title" or chk == "review title only"):
        print("Review title only")
        datasetmode=3
    else:
        print("Not valid input") #Incorrect input
        chk = ""
#Dataset alterations if specified
if(datasetmode) == 2:
    titletextdataset = dataset.copy()
    for i in range(len(dataset)):
        titletextdataset['text'][i] = dataset['title'][i] + ': ' + dataset['text'][i]
    dataset = titletextdataset  
if(datasetmode) == 3:
    titledataset = dataset.copy()
    for i in range(len(dataset)):
        titledataset['text'][i] = dataset['title'][i]
    dataset = titledataset  


#Inputs all set for this section

#now set splits. test data gets defined through here
train_eval, test_data = train_test_split(dataset, test_size=test_size, random_state=SEED) 
#and then now train + eval
train_data, eval_data = train_test_split(train_eval, test_size=eval_size, random_state=SEED) 

print("Splits are " + str(len(train_data)) + ":" + str(len(eval_data)) + ":" + str(len(test_data)))

#Create RobertaTokenizer to preprocess tokens into numerals suited for a roBERTa classifier
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_data["tokens"] = train_data.text.progress_apply(lambda x: tokenize_and_truncate(x, tokenizer,MAX_LENGTH))
eval_data["tokens"] = eval_data.text.progress_apply(lambda x: tokenize_and_truncate(x, tokenizer,MAX_LENGTH))
test_data["tokens"] = test_data.text.progress_apply(lambda x: tokenize_and_truncate(x, tokenizer,MAX_LENGTH))

#Garbage collection
torch.cuda.empty_cache()
gc.collect()

#Create dataset and dataloader 
traindataset = TextDataset(train_data["tokens"].tolist(), train_data["label"].tolist(), MAX_LENGTH)
trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True)
evaldataset = TextDataset(eval_data["tokens"].tolist(), eval_data["label"].tolist(), MAX_LENGTH)
evalloader = DataLoader(evaldataset, batch_size=BATCH_SIZE, shuffle=True)
testdataset = TextDataset(test_data["tokens"].tolist(), test_data["label"].tolist(), MAX_LENGTH)
testloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)

train_eval_dataloaders = { #only used for iteration when training is selected
    'train' : trainloader,
    'eval' : evalloader
}

model = ""

#=================User Input Section 2: Training & Evaluation==================
def eval_model(model, dataloader, epoch, loss_function):
    with torch.no_grad():
        running_loss = 0
        model.eval()
        all_predictions = []
        all_labels = []
        
        for tokens, attention_masks, labels in tqdm(dataloader):
            tokens, attention_masks, labels = tokens.to(device), attention_masks.to(device), labels.to(device)  # Move to GPU if available
            
            labels = labels.float().unsqueeze(1)  
            
            # Run inference
            outputs = model(tokens, attention_mask=attention_masks)
            # Convert outputs to predictions (binary classification)
            predictions = (outputs > 0.5).int()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loss = loss_function(outputs, labels)
            loss.requires_grad = True
            loss.backward()
            
            running_loss += loss.item() * tokens.size(0)

        # Flatten the lists and convert to numpy arrays
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()
        
        epoch_loss = running_loss / len(dataloader)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        
        print('{} epoch {}/{}:'.format('Evaluation of', epoch+1, NUM_EPOCHS))
        print('Accuracy: {:.4f}%, Loss: {:.4f} '.format(accuracy, epoch_loss))
        print('Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f} '.format(precision, recall, f1))
    return accuracy, epoch_loss


# Training function
def train_eval_model(model, dataloaders, optimizer, loss_function, device, scheduler=None):
    
    best_model = 0
    best_acc = 0
    best_epoch = 0
    noimprovement = 0

    for epoch in range(NUM_EPOCHS):
        
        model.train() # Set model to training mode
        print('Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
        print('-' * 10)
        print('Training')
        
        #Each epoch will have a training and then validation phase.
        #This will let us know how well each epoch of the model performs.
        #The best one will be stored and recorded, to be tested on the test set.

        running_loss = 0 #set running loss + accuracy values
        epoch_loss = 0
        epoch_acc = 0
        all_predictions = []
        all_labels = []

        #Start iteration
        for tokens, attention_masks, labels in tqdm(dataloaders['train']):
            tokens, attention_masks, labels = tokens.to(device), attention_masks.to(device), labels.to(device)  # Move to GPU if available
            
            #zero parameter gradients
            optimizer.zero_grad()

            # Adjust labels' shape if necessary
            labels = labels.float().unsqueeze(1)  
            
            #forward pass
            outputs = model(tokens, attention_mask=attention_masks)

            #set up preds
            predictions = (outputs > 0.5).int()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            #compute loss
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * tokens.size(0)
            

        # Flatten the lists and convert to numpy arrays
        all_predictions = np.array(all_predictions).flatten()
        all_labels = np.array(all_labels).flatten()

        epoch_loss = running_loss / len(dataloaders['train'])
        epoch_acc = accuracy_score(all_labels, all_predictions)

        #output results to tb
        print('{}: Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}'.format('Training', epoch+1, NUM_EPOCHS, epoch_loss, epoch_acc))
        
        print('Evaluating Epoch {}/{}'.format(epoch+1, NUM_EPOCHS))
        epoch_eval_acc, epoch_eval_loss = eval_model(model, dataloaders['eval'], epoch, loss_function)
        
        if epoch_eval_acc > best_acc:
            print("New best epoch accuracy.")
            best_acc = (epoch_eval_acc)
            best_epoch = epoch+1
            best_model = copy.deepcopy(model)
            noimprovement = 0
        else:
            noimprovement += 1
        if scheduler:
            scheduler.step()
        if epoch == 20:
            noimprovement = 0
        if epoch > 20:
            if noimprovement > 8:
                break
    
    print('The best epoch was {}, with {:.4f}% accuracy.'.format(best_epoch, best_acc*100))
    return best_model

#Choose pretrained or create own
chk = ""
skip = False
while chk == "":
    print("Would you like to use the best pretrained models, or to train your own?")
    print("1. Existing models")
    print("2. Custom model")
    chk = str(input("Please enter 1 or 2:")).lower() 
    if (chk == "1" or chk == "existing" or chk == "existing models" or chk == "models" or chk == "existing model"):
        print("Existing models")
        if datasetmode == 2:
            print("You chose the title text dataset earlier. Loading titletext.pt.")
            model = torch.load("modelload/titletext.pt")
        elif datasetmode == 3:
            print("You chose the title dataset earlier. Loading title.pt.")
            model = torch.load("modelload/title.pt")
        else: #datasetmode == 1
            print("You chose the text dataset earlier. Loading text.pt.")
            model = torch.load("modelload/text.pt")
        model.to(device)
        skip = True #don't need to train
    elif (chk == "2" or chk == "custom" or chk == "custom model" or chk == "custom models"):
        print("Custom model") 
        model = RobertaForBinaryClassification()
        model = nn.DataParallel(model)
        model.to(device)
    else:
        print("Not valid input") #Incorrect input
        chk = ""

#Model parameters. 
if skip:
    print("Skipping to testing...")
    reg_loss = nn.BCELoss() #most suitable for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
else:
    chk = ""
    while chk == "":
        print("Would you like to use the existing best parameters for learning rate, gamma, etc. or to enter your own parameters?")
        print("1. Existing parameters")
        print("2. Custom parameters")
        chk = str(input("Please enter 1 or 2:")).lower() 
        if (chk == "1" or chk == "existing" or chk == "existing parameters" or chk == "parameter" or chk == "existing parameter"):
            print("Existing models")
            if datasetmode == 2:
                LEARNING_RATE = 0.001
                STEP_SIZE = 12
                GAMMA = 0.1
                WEIGHT_DECAY=0.01
            elif datasetmode == 3:
                LEARNING_RATE = 0.001
                STEP_SIZE = 4
                GAMMA = 0.1
                WEIGHT_DECAY=0.01
            else: #datasetmode == 1
                LEARNING_RATE = 0.0001
                STEP_SIZE = 12
                GAMMA = 0.25
                WEIGHT_DECAY=0.002
            model.to(device)
        elif (chk == "2" or chk == "custom" or chk == "custom parameters" or chk == "custom parameter"):
            print("Custom parameters")   
            chk2 = ""   
            while chk2 == "":
                print("Please specify the value for the following:")
                print("1. Learning Rate")
                LEARNING_RATE = float(input("Please enter. Default is :"))
                print("2. Step Size")
                STEP_SIZE = float(input("Please enter.:"))
                print("3. Gamma")
                GAMMA = float(input("Please enter:"))
                print("4. Weight Decay")
                WEIGHT_DECAY = float(input("Please enter:"))
                #if all are numbers
                if not isinstance(LEARNING_RATE, numbers.Integral) and isinstance(STEP_SIZE, numbers.Integral) and isinstance(GAMMA, numbers.Integral) and isinstance(WEIGHT_DECAY, numbers.Integral):
                    print("One of these isn't a number.")
                else:
                    chk2 = "d"
        else:
            print("Not valid input") #Incorrect input
            chk = ""
    

    reg_loss = nn.BCELoss() #most suitable for binary classification
    print(LEARNING_RATE, WEIGHT_DECAY, STEP_SIZE, GAMMA)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) #before: 0.00001
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    #Start training here.
    print("Note: training is set to 100 epochs but will terminate after 30 if there have been no accuracy improvements for the last 10 epochs.")
    model = train_eval_model(model, train_eval_dataloaders, optimizer, reg_loss, device, scheduler)

#=================User Input Section 3: Testing==================
#Testing function
def acc_record(model, accuracy):
    curr_accuracy = str(round(accuracy * 100, 5))

    #get current path and file path
    curr_path = os.path.abspath(os.getcwd()) + "\\modelbest\\accuracy_reg.txt"
    print(curr_path)

    try:
        #get file size
        file_size = os.path.getsize(curr_path)

        #if file size is 0, it's empty. populate it with accuracy value
        if (file_size == 0):
            t_file = open(curr_path, 'w+')
            t_file.write(curr_accuracy)
            t_file.close()

            torch.save(model, "modelbest/modelbest_reg.pt")
            print("File is empty.")
            print("Model saved as no prior best exists.")
        else:
            t_file = open(curr_path, 'r+')
            best_acc = t_file.read()
            if float(curr_accuracy) > float(best_acc):
                t_file = open(curr_path, 'w+')
                t_file.write(curr_accuracy)
                t_file.close()

                torch.save(model, "modelbest/modelbest_reg.pt")

                print("The best recorded accuracy is " + best_acc +"%.")
                print("The accuracy of this model is " + curr_accuracy + "%.")
                print("Model saved as new best has been achieved.")
            else:
                print("The best recorded accuracy is " + best_acc +"%.")
                print("The accuracy of this model is " + curr_accuracy + "%.")
                print("Model not saved as new best not achieved.")
            t_file.close()
    except FileNotFoundError as e:
        #file doesn't exist. Create it
        t_file = open(curr_path, 'w+')
        t_file.write(curr_accuracy)
        t_file.close()

        torch.save(model, "modelbest/modelbest_reg.pt")
        print("Accuracy file did not exist. Writing to new .txt file.")
        print("Model saved as no prior best exists.")

def test_model(model, testloader, testdata, device, record=True):
    # Put model in evaluation mode
    model.eval()

    # Initialize lists to store true labels and predictions
    predictions, true_labels = [],[]
    with torch.no_grad():
        for tokens, attention_masks, labels in tqdm(testloader):
            tokens, attention_masks, labels = tokens.to(device), attention_masks.to(device), labels.to(device)

            # Run inference
            outputs = model(tokens, attention_mask=attention_masks)

            logits = outputs.detach().cpu().numpy()
            predictions.append(logits)

    # Flatten the predictions and true values for aggregate evaluation on the whole dataset
    predictions = np.concatenate(predictions, axis=0)
    true_labels = testdata["label"].values
    true_labels_list = [round(float(x)) for x in true_labels]

    # For each sample, pick the label (0 or 1) with the higher score as our prediction.
    predslist = [round(float(x)) for x in predictions]
    preds = np.array(predslist)

    # Ensure preds and true_labels are the same shape
    assert preds.shape == true_labels.shape

    # Calculate the accuracy rate
    accuracy = (preds == true_labels).mean()

    # Calculate precision, recall, f1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_list, predslist, average='binary')

    # Calculate AUC-ROC
    roc_auc = roc_auc_score(true_labels_list, predslist)

    print(f'Prediction: {preds}')
    print(f'Actual Labels: {true_labels}')
    print(f'Accuracy: {round(accuracy * 100,5)}%')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'ROC-AUC: {roc_auc}')
    
    if (record):
        acc_record(model, accuracy)
    return true_labels_list, predslist



print("Starting testing.")
reg_true_labels_list, reg_predslist = test_model(model, testloader, test_data, device)
str(input("Final results. Press enter to quit."))
raise SystemExit(0)

