import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pdfminer.high_level import extract_text

from src.config import Config

def main():
    print("Reading Raw PDF Files:")
    print("  extracting all the files inside PDF directory...")
    pdf_files = os.listdir(Config.FILES["PDF_DIR"])

    print("  putting read PDF files into dictionary...")
    text_dict = {}
    for file in pdf_files:
        read_pdf_file = os.path.join(Config.FILES["PDF_DIR"], file)
        text = extract_text(read_pdf_file)
        text_dict[file] = text

    print("  create PDF dataframe from PDF dictionary...")
    pdf_df = pd.DataFrame(list(text_dict.items()), columns=["file_path", "raw_text"])

    print("  saving dataframe into local DATABASE: {}...".format(Config.FILES["DATABASE_DIR"]))
    engine = create_engine("sqlite:///" + Config.FILES["DATABASE_DIR"])
    df.to_sql("Text_table", engine, if_exists="replace", index=False)

    print("  done extracting PDF data...")

if __name__ == "__main__":
    main()
