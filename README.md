### Description:
News-Analyzer is a classical machine learning model that can determine with high accuracy more than 90% whether the news is real (REAL) or FAKE (FAKE).

When developing the model, TfidfVectorizer was used to extract features from text data and PassiveAggressiveClassifier.


### Installation:
- download the project
- make install
- Register it in the file .env the PATH_TO_FILE variable, which specifies the path to the dataset

### Run:
- make run

### Visualization:
#### Let's analyze the data set and build a classic machine learning model that can determine with more than 90% accuracy whether the news is real or fake. Visualize the results and generate a report.

#### Downloading and analyzing the dataset
- looking through the first ten entries

[![image.png](https://i.postimg.cc/sgC885d1/image.png)](https://postimg.cc/hfMpQzGq)

- studying the data structure

[![image.png](https://i.postimg.cc/x1G7Sqn5/image.png)](https://postimg.cc/Wd3SGNDq)

- check for any null values

[![image.png](https://i.postimg.cc/LsHW4Rsp/image.png)](https://postimg.cc/pm6Ckw8G)

#### Visualization:
- the ratio of labels, REAL and FAKE

[![1.png](https://i.postimg.cc/3NP63vkS/1.png)](https://postimg.cc/0K0ZfrdD)

- text length distributions in the dataset: REAL and FAKE

[![2.png](https://i.postimg.cc/65vgYYPD/2.png)](https://postimg.cc/gnGg0qmD)

#### Building a classical machine learning model using TfidfVectorizer and PassiveAggressiveClassifier, predicting on a test set and calculating accuracy
```
10 important words for text classification:
said: 4.73
says: 3.27
marriage: 2.77
conservative: 2.75
friday: 2.64
gop: 2.51
tuesday: 2.39
rush: 2.37
march: 2.29
sanders: 2.25
```

```
Accuracy based on test data: 92.98%
```

#### Building an сonfusion matrix

[![u4.png](https://i.postimg.cc/wxPnXSwX/u4.png)](https://postimg.cc/V5q71HQv)

#### Classification report

[![jnxth.png](https://i.postimg.cc/VvSpyRk0/jnxth.png)](https://postimg.cc/hQBpTVxK)
