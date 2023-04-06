import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
import numpy as np

from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from src.config import Config


class Logger(object):
    info = print
    warning = print
    critical = print
    error = print
    

class Analysis(Config):
    data = {}
    
    def __init__(self, suffix:str="", logger=Logger()):
        self.suffix = suffix
        self.logger = logger
        
        
    def reading_raw_news(self, tokenizer):
        self.logger.info("Reading Raw News Data:")
        self.data["news_summary_df"] = pd.read_csv(os.path.join(self.FILES["DATA_LOCAL_DIR"], "{}.csv".format(self.FILES["RAW_NEWS_SUMMARY_FILE"])), encoding="latin-1")
        
        self.logger.info("  extract only the 'text' and 'ctext' from news summary dataframe...")
        self.data["text_df"] = self.data["news_summary_df"][["text", "ctext"]]
        self.data["text_df"].columns = ["summary", "text"]
        self.data["text_df"] = self.data["text_df"].dropna()
        
        self.logger.info("  create train and test data from the raw news summary dataframe...")
        self.data["train_df"], self.data["test_df"] = train_test_split(self.data["text_df"], test_size=self.ANALYSIS_CONFIG["SPLIT_RATIO"])
        
        self.logger.info("  calculating the counts of tokens of corpus from train data...")
        self.text_token_counts, self.summary_token_counts = [], []
        for _, row in self.data["train_df"].iterrows():
            self.text_token_count = len(tokenizer.encode(row["text"]))
            self.text_token_counts.append(self.text_token_count)
            
            self.summary_token_count = len(tokenizer.encode(row["summary"]))
            self.summary_token_counts.append(self.summary_token_count)
        
        self.logger.info("  done.")
        
        
    def multi_histogram_plots(self, xvar1, xvar2, title1, title2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
        sns.histplot(xvar1, ax=ax1)
        ax1.set_title(title1)

        sns.histplot(xvar2, ax=ax2)
        ax2.set_title(title2)

        return plt.show()
        
        
class NewSummaryDataset(Dataset):
    def __init__(
        self, 
        data:pd.DataFrame(), 
        tokenizer:T5Tokenizer, 
        text_max_token_len:int=Config.ANALYSIS_CONFIG["TEXT_MAX_TOKEN_LEN"], 
        summary_max_token_len:int=Config.ANALYSIS_CONFIG["SUMMARY_MAX_TOKEN_LEN"]
        ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem(self, index:int):
        data_row = self.data.iloc[index]
        text = data_row["text"]
        
        text_encoding = tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding=Config.ANALYSIS_CONFIG["TOKENIZER_PADDING"],
            truncation=Config.ANALYSIS_CONFIG["TOKENIZER_TRUNCATION"],
            return_attention_mask=Config.ANALYSIS_CONFIG["TOKENIZER_ATTENTION_MASK"],
            add_special_token=Config.ANALYSIS_CONFIG["TOKENIZER_ADD_SPECIAL_TOKEN"],
            return_tensors=Config.ANALYSIS_CONFIG["TOKENIZER_RETURN_TENSORS"],
        )
        
        summary_encoding = tokenizer(
            data_row["summary"],
            max_length=self.summary_max_token_len,
            padding=Config.ANALYSIS_CONFIG["TOKENIZER_PADDING"],
            truncation=Config.ANALYSIS_CONFIG["TOKENIZER_TRUNCATION"],
            return_attention_mask=Config.ANALYSIS_CONFIG["TOKENIZER_ATTENTION_MASK"],
            add_special_token=Config.ANALYSIS_CONFIG["TOKENIZER_ADD_SPECIAL_TOKEN"],
            return_tensors=Config.ANALYSIS_CONFIG["TOKENIZER_RETURN_TENSORS"],
        )
        
        labels = summary_encoding["input_ids"]
        labels[labels==0] = Config.ANALYSIS_CONFIG["ZERO_INPUT_IDS"]
        
        return dict(
            text=text,
            summary=data_row["summary"],
            text_input_ids=text_encoding["input_ids"].flatten(),
            text_attention_mask=text_encoding["attention_mask"].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding["attention_mask"].flatten()
        )
        
        
class NewsSummaryDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_df:pd.DataFrame, 
        test_df:pd.DataFrame, 
        tokenizer:T5Tokenizer, 
        batch_size:int=8, 
        text_max_token_len:int=512,
        summary_max_token_len:int=128
    ):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        
        
    def setup(self, stage=None):
        self.train_dataset = NewSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        
        self.test_dataset = NewSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=16
        )
    
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16
        )
    
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=16
        )
        
        
