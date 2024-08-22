import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def visualize_ratio(df):
    plt.figure(figsize=(8, 6))
    plt.pie(df['label'].value_counts(), labels=df['label'].value_counts().index,
            autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Label ratio REAL and FAKE')
    plt.show()


def visualize_length(df):
    df['text_len'] = df['text'].apply(len)
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='text_len', hue='label', bins=30,
                 palette=['skyblue', 'salmon'])
    plt.title('Distribution of text length by label')
    plt.xlabel('Text Length')
    plt.ylabel('Count')
    plt.show()


def visualize_confusion_matrix(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    plt.figure(figsize=(8, 6))
    axis_labels = ['FAKE', 'REAL']
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=axis_labels, yticklabels=axis_labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
