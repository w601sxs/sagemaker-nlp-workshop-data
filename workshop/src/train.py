
# Importing stock ml libraries
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertModel
import logging
logging.basicConfig(level=logging.ERROR)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

import random
import logging
import sys
import argparse
import os

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }
    
class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 9)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Uncomment to debug GPU stuff
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Parse arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--input_file", type=str, default="sagemaker_input_hf.json")
    parser.add_argument("--model_id", type=str)
    
    args, _ = parser.parse_known_args()
    
    tokenizer = DistilBertTokenizer.from_pretrained(args.model_id, truncation=True, do_lower_case=True)
    
    
    print(os.path.join(os.environ["SM_CHANNEL_TRAIN"], args.input_file))
    data = pd.read_json(os.path.join(os.environ["SM_CHANNEL_TRAIN"], args.input_file), orient='record', lines=True)
    #rename and move
    data.columns = ['labels','text']
    data = data[['text','labels']]
    
    
    new_df = data.copy()
    train_size = 0.8
    train_data=new_df.sample(frac=train_size,random_state=200)
    test_data=new_df.drop(train_data.index).reset_index(drop=True)
    train_data = train_data.reset_index(drop=True)


    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_data.shape))
    print("TEST Dataset: {}".format(test_data.shape))

    training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
    testing_set = MultiLabelDataset(test_data, tokenizer, MAX_LEN)
    
    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    

    model = DistilBERTClass()
    model.to(device)
    
    def loss_fn(outputs, targets):
        return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=float(args.learning_rate))
    
    def train(epoch):
        model.train()

        for _,data in tqdm(enumerate(training_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)

            outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            if _%5000==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    
    for epoch in range(args.epochs):
        train(epoch)
    
    # TODO - eval model on test set
    
    output_model_file = os.path.join(os.environ["SM_MODEL_DIR"],'pytorch_distilbert_news.bin')
    output_vocab_file = os.path.join(os.environ["SM_MODEL_DIR"],'vocab_distilbert_news.bin')

    torch.save(model, output_model_file)
    tokenizer.save_vocabulary(output_vocab_file)

    print('Model Saved')
