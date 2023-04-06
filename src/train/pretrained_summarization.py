import time
import pandas as pd
from sqlalchemy import create_engine

import torch
from summarizer import Summarizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import Config

class Logger(object):
    info = print
    warning = print
    critical = print
    error = print


class PretrainedSummarization(Config):
    def __init__(self, logger=Logger()):
        self.logger = logger


    def ext_sum(self, text:str, ratio:float=0.8):
        """
        Generate extractive summary using BERT model

        INPUT:
        text - str. Input text
        ratio - float. Enter a ratio between 0.1 - 1.0 [default = 0.8]
                (ratio = summary length / original text length)

        OUTPUT:
        summary - str. Generated summary
        """
        bert_model = Summarizer()
        summary = bert_model(text, ratio=ratio)

        return summary
    

    def abs_sum(self, text:str, model, tokenizer, min_length:int=80,
                max_length:int=150, length_penalty:int=15,
                num_beams:int=2):
        """
        Generate abstractive summary using T5 model

        INPUT:
        text - str. Input text
        model - model name
        tokenizer - model tokenizer
        min_length - int. The min length of the sequence to be generated
                        [default = 80]
        max_length - int. The max length of the sequence to be generated
                        [default = 150]
        length_penalty - float. Set to values < 1.0 in order to encourage the model
                        to generate shorter sequences, to a value > 1.0 in order to
                        encourage the model to produce longer sequences.
                        [default = 15]
        num_beams - int. Number of beams for beam search. 1 means no beam search
                        [default = 2]

        OUTPUT:
        summary - str. Generated summary
        """
        tokens_input = tokenizer.encode("summarize: "+text, return_tensors='pt',
                                        # model tokens max input length
                                        max_length=tokenizer.model_max_length,
                                        truncation=True)

        summary_ids = model.generate(tokens_input,
                                    min_length=min_length,
                                    max_length=max_length,
                                    length_penalty=length_penalty,
                                    num_beams=num_beams)

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


    def generate_summary(self, text:str, model, tokenizer, ext_ratio:float=1.0, min_length:int=80,
                        max_length:int=150, length_penalty:int=15,
                        num_beams:int=2):
        """
        Generate summary for using extractive & abstractive methods

        INPUT:
        text - str. Input text
        model - model name
        tokenizer - model tokenizer
        ext_ratio - float. Enter a ratio between 0.1 - 1.0 [default = 1.0]
                    (ratio = summary length / original text length)
                    1.0 means no extractive summarization is performed before
                    abstractive summarization
        min_length - int. The min length of the sequence to be generated
                    [default = 80]
        max_length - int. The max length of the sequence to be generated
                    [default = 150]
        length_penalty - float. Set to values < 1.0 in order to encourage the model
                        to generate shorter sequences, to a value > 1.0 in order to
                        encourage the model to produce longer sequences.
                        [default = 15]
        num_beams - int. Number of beams for beam search. 1 means no beam search
                        [default = 2]

        OUTPUT:
        summary - str. Generated summary
        """
        if ext_ratio == 1.0:
            summary = abs_sum(text, model, tokenizer, min_length, max_length, length_penalty, num_beams)
        elif ext_ratio < 1.0:
            text = ext_sum(text, ratio=ext_ratio)
            summary = abs_sum(text, model, tokenizer, min_length, max_length, length_penalty, num_beams)
        else:
            self.logger.info("error, please enter value for ext_ratio between 0-1.0")
            exit()

        return summary


    def gen_sum_save_monitor(self, df, model, tokenizer, output_folder, ext_ratio=1.0,
                            min_length=80, max_length=150, length_penalty=15,
                            num_beams=2):
        """
        Monitor progress while generating summary & save output to list & text file

        INPUT:
        df - DataFrama. Data loaded from database
        model - model name
        tokenizer - model tokenizer
        output_folder - str. Folder name to save the generated output in text file
        ext_ratio - float. Enter a ratio between 0.1 - 1.0 [default = 1.0]
                    (ratio = summary length / original text length)
                    1.0 means no extractive summarization is performed before
                    abstractive summarization
        min_length - int. The min length of the sequence to be generated
                    [default = 80]
        max_length - int. The max length of the sequence to be generated
                    [default = 150]
        length_penalty - float. Set to values < 1.0 in order to encourage the model
                        to generate shorter sequences, to a value > 1.0 in order to
                        encourage the model to produce longer sequences.
                        [default = 15]
        num_beams - int. Number of beams for beam search. 1 means no beam search
                    [default = 2]

        OUTPUT:
        summaries - list. Generated summary appended to a list
        """
        summaries = []
        for i in range(len(df)):
            file_path = df.file_path[i]
            raw_text = df.raw_text[i]

            start = time.time()
            summary = generate_summary(raw_text, model, tokenizer, ext_ratio, 
                                       min_length, max_length, length_penalty, num_beams)

            file_name = file_path[8:][:-4] + "_summary.txt"

            with open(output_folder + "/" + file_name, "w")as text_file:
                text_file.write(summary)

            summaries.append(summary)
            end = time.time()
            print("Summarized '{}'[time: {:.2f}s]".format(file_path, end-start))

        return summaries


    def main(self):
        self.logger.info("Begin Pretrained T-5 Summarization Model:")
        self.logger.info("  loading data from database...")
        engine = create_engine("sqlite:///" + Config.FILES["DATABASE_DIR"])
        df = pd.read_sql_table("Text_table", engine)

        self.logger.info("  loading summarization & tokenization model: {}".format(self.MODELLING_CONFIG["MODEL_NAME"]))
        model = AutoModelForSeq2SeqLM.from_pretrained(self.MODELLING_CONFIG["MODEL_NAME"])
        tokenizer = AutoTokenizer.from_pretrained(self.MODELLING_CONFIG["MODEL_NAME"])

        self.logger.info("Generating Summary:")
        summaries = gen_sum_save_monitor(df, model, tokenizer, output_folder=self.MODELLING_CONFIG["PREPROCESS_DIR"],
                                         ext_ratio=self.MODELLING_CONFIG["EXT_RATIO"], 
                                         min_length=self.MODELLING_CONFIG["MIN_LENGTH"],
                                         max_length=self.MODELLING_CONFIG["MAX_LENGTH"], 
                                         length_penalty=self.MODELLING_CONFIG["LENGTH_PENALTY"],
                                         num_beams=self.MODELLING_CONFIG["NUM_BEAMS"])

        df["summary"] = summaries

        self.logger.info("  saving dataframe into local DATABASE: {}...".format(Config.FILES["DATABASE_DIR"]))
        engine = create_engine("sqlite:///" + Config.FILES["DATABASE_DIR"])
        df.to_sql("Text_table", engine, if_exists="replace", index=False)

        self.logger.info("  done performing pre-trained summarization...")


if __name__ == '__main__':
    summarizer = PretrainedSummarization()
    summarizer.main()
