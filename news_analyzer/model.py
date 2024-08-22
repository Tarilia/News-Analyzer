from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (accuracy_score,
                             classification_report)

from news_analyzer.visualization import visualize_confusion_matrix


def train_the_model(x_train, x_test, y_train, y_test):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(x_train)
    tfidf_test = tfidf_vectorizer.transform(x_test)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_features = sorted(zip(pac.coef_[0], feature_names), reverse=True)[:10]

    print('\n10 important words for text classification:')
    for coef, feat in top_features:
        print(f'{feat}: {coef:.2f}')

    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(f'\nCalculating the accuracy: {score:.2f}')

    print('\nVisualize confusion matrix')
    visualize_confusion_matrix(y_test, y_pred)
    print(f'\nWe are displaying the report:'
          f' \n{classification_report(y_test, y_pred)}')
