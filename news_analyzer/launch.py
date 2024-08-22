import os
from dotenv import load_dotenv

import pandas as pd
from sklearn.model_selection import train_test_split

from news_analyzer.analysis import analyze_the_data
from news_analyzer.model import train_the_model
from news_analyzer.visualization import visualize_ratio, visualize_length

load_dotenv()

PATH_TO_FILE = os.getenv('PATH_TO_FILE')
df = pd.read_csv(PATH_TO_FILE)


def to_run():
    analyze_the_data(df)
    print('\nWe visualize the ratio of labels, REAL and FAKE')
    visualize_ratio(df)
    print('\nVisualize text length distributions in the dataset: REAL and FAKE')
    visualize_length(df)
    x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'],
                                                        test_size=0.2,
                                                        random_state=7)
    train_the_model(x_train, x_test, y_train, y_test)
