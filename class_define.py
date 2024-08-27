import random

from tqdm.auto import tqdm
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


class RobertaForBinaryClassification(nn.Module):
    def __init__(self):
        super(RobertaForBinaryClassification, self).__init__()

        self.roberta = RobertaModel.from_pretrained("roberta-base")
        hidden_size = self.roberta.config.hidden_size

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        x = self.fc1(cls_output)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        probs = self.sigmoid(logits)
        return probs

class TextDataset(Dataset):
    def __init__(self, token_lists, labels, max_token_len):
        self.token_lists = token_lists
        self.labels = labels
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.token_lists)

    def __getitem__(self, idx):
        # Pad the token lists to the max length
        token_list = self.token_lists[idx][:self.max_token_len]  # truncate if longer than max length
        padded_tokens = token_list + [0] * (self.max_token_len - len(token_list))  # pad with zeros
        attention_mask = [1 if i < len(token_list) else 0 for i in range(self.max_token_len)]
        return torch.tensor(padded_tokens), torch.tensor(attention_mask), torch.tensor(self.labels[idx])