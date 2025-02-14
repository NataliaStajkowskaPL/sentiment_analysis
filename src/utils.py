# src/utils.py

import os
import pandas as pd

def load_data(data_dir):
    """
    Loads data from a folder containing text files in the 'pos' and 'neg' subdirectories.  
    :param data_dir: Path to the data folder.  
    :return: DataFrame containing the data (text and labels).  
    """
    texts = []
    labels = []
    
    # Load data from the 'pos' folder
    pos_dir = os.path.join(data_dir, 'pos')
    for filename in os.listdir(pos_dir):
        with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append('pos')  # pos labels
    
    # Load data from the 'neg' folder
    neg_dir = os.path.join(data_dir, 'neg')
    for filename in os.listdir(neg_dir):
        with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append('neg')  # neg labels
    
    # Create a DataFrame with texts and labels
    data = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    return data
