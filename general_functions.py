import random

from tqdm.auto import tqdm
from transformers import RobertaTokenizer
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def dataset_load(path, strip_instructions=[], encodingin='cp1252'):
    data=pd.read_csv(path, encoding=encodingin) #read dataset CSV from path given
    
    #Iterate and strip unwanted columns from dataset
    for column in strip_instructions:
        del data[column]
    
    return data

def dataset_label_convert(dataset, label_instructions, column='label'):
    #Iterate and convert dataset labels according to instruction dict given
    for i in range(len(dataset)):
        dataset['label'][i] = label_instructions[dataset[column][i].lower()]
    return dataset

def tokenize_and_truncate(text, tokenizer, max_length=256):
    # Tokenize and truncate
    tokens = tokenizer.tokenize(text)[:max_length]  # leaving space for special tokens
    # Add the special tokens
    
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return token_ids

#ACCURACY FOR EXPERIMENT_REVIEW DATASET.
#NEED TO RETURN:
    #1. Accuracy of subject OVERALL.
    #2. Accuracy of subject for human reviews.
    #3. Accuracy of subject for AI reviews.
    
    #4. Accuracy of subject when it came to 200 Salminen entries.
    #5. Accuracy of subject for Salminen human reviews.
    #6. Accuracy of subject for Salminen AI (GPT-2) reviews.
    
    #7. Accuracy of subject when it came to 200 FRDDS entries.
    #8. Accuracy of subject for FRDDS human reviews.
    #9. Accuracy of subject for FRDDS AI (GPT-4o) reviews. 

def all_accuracy(preds, data, notmodel=True):
    overall_correct = 0
    overall_human_correct = 0
    overall_ai_correct = 0
    
    overall_salminen_correct = 0
    overall_salminen_human_correct = 0
    overall_salminen_ai_correct = 0
    
    overall_frdds_correct = 0
    overall_frdds_human_correct = 0
    overall_frdds_ai_correct = 0
    
    if (notmodel):
        preds = preds["label"].to_list()
    
    for i in range(len(data)):
        if preds[i] == data["label"][i]:
            overall_correct += 1
            if data["label"][i] == 1:
                overall_human_correct += 1
                if data["dataset"][i] == "Salminen":
                    overall_salminen_human_correct += 1
                    overall_salminen_correct += 1
                if data["dataset"][i] == "FRDDS":
                    overall_frdds_human_correct += 1
                    overall_frdds_correct += 1
            if data["label"][i] == 0:
                overall_ai_correct += 1
                if data["dataset"][i] == "Salminen":
                    overall_salminen_ai_correct += 1
                    overall_salminen_correct += 1
                if data["dataset"][i] == "FRDDS":
                    overall_frdds_ai_correct += 1
                    overall_frdds_correct += 1

    overall_accuracy = (overall_correct / len(data)) * 100 #1
    overall_human_accuracy = (overall_human_correct / (len(data) / 2)) * 100 #2
    overall_ai_accuracy = (overall_ai_correct / (len(data) / 2)) * 100 #3
    
    overall_salminen_accuracy = (overall_salminen_correct / (len(data) / 2)) * 100 #4
    overall_salminen_human_accuracy = (overall_salminen_human_correct / (len(data) / 4)) * 100 #5
    overall_salminen_ai_accuracy = (overall_salminen_ai_correct / (len(data) / 4)) * 100 #6
    
    overall_frdds_accuracy = (overall_frdds_correct / (len(data) / 2)) * 100 #7
    overall_frdds_human_accuracy = (overall_frdds_human_correct / (len(data) / 4)) * 100 #8
    overall_frdds_ai_accuracy = (overall_frdds_ai_correct / (len(data) / 4)) * 100 #9

    
    
    return [overall_accuracy, overall_human_accuracy, overall_ai_accuracy, overall_salminen_accuracy, overall_salminen_human_accuracy, overall_salminen_ai_accuracy, overall_frdds_accuracy, overall_frdds_human_accuracy, overall_frdds_ai_accuracy]