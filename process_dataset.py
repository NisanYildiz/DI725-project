from random import randrange
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

def process_df_captions(captions: pd.DataFrame, path: str):

  train_jsonl = path + "/train_data.jsonl"
  val_jsonl = path + "/val_data.jsonl"
  test_jsonl = path + "/test_data.jsonl"

  if os.path.exists(train_jsonl):
    os.remove(train_jsonl)

  if os.path.exists(val_jsonl):
    os.remove(val_jsonl)

  if os.path.exists(test_jsonl):
    os.remove(test_jsonl)

  for img_i in tqdm(range(len(captions)), desc="Processing captions"):
    match captions.iloc[img_i, 1]:
      case 'train':
        file = open(train_jsonl, "a")
      case 'val':
        file = open(val_jsonl, "a")
      case 'test':
        file = open(test_jsonl, "a")

    ## randomly pick a caption each time
    caption_index = randrange(3,8)
    caption = captions.iloc[img_i, caption_index]

    if '\n' in caption:
      caption = caption.replace('\n', ' ')

    line = f'{{"file_name" : "{captions.iloc[img_i, 2]}", "prefix": "caption en", "suffix": "{caption}"}}'
    file.write(line + "\n")
  file.close()

if __name__ == "__main__":
    captions = pd.read_csv("RISCM/captions.csv")
    process_df_captions(captions, path="RISCM/resized")
    print("Dataset processing completed.")
    
