import os
import torch
import json
import math
import spacy
import pickle
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator
from collections import defaultdict
import datasets
import nltk
from nltk.corpus import stopwords
import re
class WikiDataset(Dataset):
    def __init__(self, file_path: str, mode: str, 
                 model, tokenizer, max_length=32, accelerator = None):
        
        
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.accelerator = accelerator or Accelerator()

        if "llara_first" in mode:
            self.lines = datasets.load_dataset(
                file_path, 
                split='train'
            )
            required_fields = ['input']
            for field in required_fields:
                if field not in self.lines.features:
                    raise ValueError(f"Dataset missing required field: {field}")
            self.lines = self.lines['input']
            self.lines = [line for line in self.lines if line.strip()]
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.lines = file.readlines()
            
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):

        text = self.lines[idx].strip()
        if self.mode == "llara_original_style":
            prefix = '"'
            suffix = '", summarize the above passage within eight words: <s1><s2><s3><s4><s5><s6><s7><s8>'
            prefix_ids = self.tokenizer(prefix, truncation=True, max_length=self.max_length, return_tensors=None)['input_ids']
            suffix_ids = self.tokenizer(suffix, truncation=True, max_length=self.max_length, return_tensors=None, add_special_tokens=False)['input_ids']
        
            inputs = self.tokenizer(text,
                                   truncation=True,
                                   max_length=self.max_length - len(prefix_ids) - len(suffix_ids),
                                   padding=False,
                                   return_tensors=None,
                                   add_special_tokens=False)

            input_ids = prefix_ids + inputs['input_ids'] + suffix_ids
            attention_mask = [1] * len(input_ids)
                
            encoding = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            current_length = len(input_ids)
            
            if current_length >= self.max_length:
                encoding['input_ids'] = input_ids[-self.max_length:]
                encoding['attention_mask'] = attention_mask[-self.max_length:]
                
            else:
                pad_length = self.max_length - current_length
                pad_token_id = getattr(self.tokenizer, 'pad_token_id', 2)  

                encoding['input_ids'] = [pad_token_id] * pad_length + input_ids
                encoding['attention_mask'] = [0] * pad_length + attention_mask
                
            encoding["line_text"] = text  
            return encoding
            
        
        if self.mode == "prompt_lastpool_lmhead" or "last" in self.mode:
            truncated_sentence = self.model._prompt_and_tokenize(text)

        elif self.mode == "prompt_attention":
            truncated_sentence = self.model._prompt_and_tokenize(text)
        
        elif self.mode == "noprompt_meanpool_lmhead" or "mean" in self.mode:
            truncated_sentence = text  
            
        encoding = self.tokenizer(
            truncated_sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["line_text"] = text  
        
        
            
        if "llara_first" in self.mode:
            # nltk.download('stopwords', quiet=True)
            stop_words = stopwords.words('english')
            stop_words.extend(['!', ',' ,'.' ,'?'])
        
            text = re.sub(r'[\W_]+', ' ', text)
            text = ' '.join([word for word in text.split() if word.lower() not in stop_words])

            target = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            target_ids = target['input_ids']
            target_mask = target['attention_mask']
            
            target_ids = target_ids[target_mask == 1]
            unique_ids = torch.unique(target_ids[(target_ids > 2) & (target_ids != 28705)]).detach()

            encoding["unique_id"] = unique_ids.detach().clone()
            encoding["line_text"] = text  
            
        return encoding
        