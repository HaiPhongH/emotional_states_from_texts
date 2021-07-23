import os
import torch
import numpy as np
import underthesea
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from sklearn.svm import SVC
import pandas as pd
from joblib import dump
import re

class TrainingBert():

    # load pre-trained PhoBERT model
    def load_bert_model(self):
        phobert = AutoModel.from_pretrained('vinai/phobert-base')
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base', use_fast=False)
        return phobert, tokenizer

    # standardize input data
    def standardize_data(self, row):
        # remove , . ? at the end of a sentence
        row = re.sub(r"[\.,\?]+$-", "", row)

        # remove all of , . ; “ : ” " ' ! ? - in a sentence
        row = row.replace(",", " ").replace(".", " ") \
            .replace(";", " ").replace("“", " ") \
            .replace(":", " ").replace("”", " ") \
            .replace('"', " ").replace("'", " ") \
            .replace("!", " ").replace("?", " ") \
            .replace("-", " ").replace("^", " ") \
            .replace("*", " ").replace("#", " ") \
            .replace("%", " ").replace("@", " ") \
            .replace("(", " ").replace(")", " ") \
            .replace("$", " ").replace("~", " ") \
            .replace("`", " ").replace("|", " ") \
            .replace("/", " ").replace("+", " ") \
            .replace("*", " ").replace("\\", " ") \
            .replace("<", " ").replace(">", " ")
        row = row.strip().lower()
        return row

    # load the list of stop words in Vietnamese - the words contain no useful information
    def load_stopwords(self):
        sw = []
        with open("data/stopwords.txt", encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            sw.append(line.replace("\n",""))
        return sw
    
    # load data from csv file - the file contains a list of review for the product
    def load_data(self, filename):
        list_review = []
        list_label = []

        with open(filename, encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            line = line.replace("\n", "")
            list_review.append(self.standardize_data(line[:-2]))
            list_label.append(int(line[-1].replace("\n","")))
        
        return list_review, list_label

    # create feature vectors
    def create_feature_vector(self, list_review):
        phobert, tokenizer = self.load_bert_model()
        stopwords = self.load_stopwords()
        tokenizered = []
        
        # for each review in list review
        for review in list_review:
            # tokenize sentences in to words
            line = underthesea.word_tokenize(review)

            # only take the useful words - remove all the stop words
            filtered_line = [word for word in line if not word in stopwords]

            # put all the filtered words into a complete sentence
            line = " ".join(filtered_line)
            line = underthesea.word_tokenize(line, format='text')
            
            #tokenize by PhoBERT
            line = tokenizer.encode(line)
            tokenizered.append(line)
        
        # find the maximum value of review's length in the all dataset
        max_len = 0
        for value in tokenizered:
            if len(value) > max_len:
                max_len = len(value)
        
        # for the review which has length smaller than max_len, set all the rest value is 0
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenizered])

        # while extract feature vector, do not perform on the word has the value is 0
        attention_mask = np.where(padded == 0, 0, 1)

        # convert to tensor
        padded = torch.tensor(padded)
        attention_mask = torch.tensor(attention_mask)
        with torch.no_grad():
            print("here")
            last_hidden_states = phobert(padded, attention_mask = attention_mask)
        
        # get features from PhoBERT
        features = last_hidden_states[0][:, 0, :].numpy()
        print(features.shape)

        return features
