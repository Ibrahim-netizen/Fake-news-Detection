# Fake-news-Detection
This project investigate the analysis and results obtained from the classification of real and fake news with dataset from Kaggle with logistic regression machine learning technique and Term-frequency Inverse document frequency (TF-IDF) vectorizer, designed to handle categorical data with several performance metrics including model accuracy score, confusion matrix, sensitivity, specificity, F1-score, precision and recall as well as receiver operating characteristics (ROC) curve. Results and accuracy score shows a 97% F1 score of proportions of successfully classified news article, 0.97 ROC AUC score showing the high efficacy of the regression model. This shows the high fitness of logistic regression model for binary classification and accuracy of this research.

## Brief Background
Fake news dissemination is a persistent dilemma that continues to plague the news channels and media ecosystem in general and requires solution in tackling falsification of information and misleading claims with the capability of posing a threat to data authenticity and the public at large. This Project reports the findings of a study that classified real and fake news using a Kaggle dataset and a Term-frequency Inverse Document Frequency (TF-IDF) vectorizer, which is designed to handle categorical data and includes several performance metrics such as model accuracy score, confusion matrix, sensitivity, specificity, and F1-score. The annotated dataset integrated in this research is an ideal candidate for this investigation. The machine learning algorithm incorporated into this research confirms its adaptivity with binary distribution dataset. This report also shows various data visualization and performance metrics obtained from the detection of fake news with binary text classification technique.

## Fake News Dataset
The dataset employed for this research was identified after rigorous research on the internet. Finally, a wellannotated dataset on Kaggle was identified initially uploaded for prediction competition themed “Fake News” with the objective of building a machine learning model to identify unreliable news article on the web-based environment. The integrated dataset was employed in the experimental research of fake news detection with logistic regression algorithm was retrieved from Kaggle, an open-source machine learning 
community that allows users to discover and upload datasets in an internet environment to build machine learning models. The dataset named Fake news dataset consisted of 3 distinct datasets of news articles comprising of 5 features was retrieved on the 11/06/2021 from https://www.kaggle.com/c/fake-news/data . The dataset named train.csv was collected consisting of 20800 entries which is quantifiable for the purpose of this research analysis. The dataset consists of mixed datasets which includes discrete and continuous variable data types. The missed dataset has a storage memory of 812.6kilobytes with two integer and three object data types.

![image](https://user-images.githubusercontent.com/76513466/137642024-33fe626e-3a49-45a5-8e8c-715d29f109ed.png)

## Logistic Regression
A strong statistical method for modelling a binomial result with one or more explanatory factors is logistic regression. It estimates probabilities using a logistic function, which is the cumulative logistic distribution, to quantify the relationship between the categorical dependent variable and one or more independent variables. When the value of the target variable is categorical in nature, logistic regression is implemented as a classification algorithm. Unlike linear regression, which produces continuous numerical values, logistic regression produces a probability value that may be mapped to two or more discrete classes using the logistic sigmoid function. Logistic regression is a classification algorithm, used when the value of the target variable is categorical in nature.

![image](https://user-images.githubusercontent.com/76513466/137642142-0467bead-54bc-432c-bd0b-87eacd8ffdd6.png)

When plotted on a graph, the sigmoid function appears to be a "S" shaped curve. It takes numbers between 0 and 1 and "squishes" them towards the top and bottom boundaries, identifying them as 0 or 1. The equation of the sigmoid function is denoted as follows:
![image](https://user-images.githubusercontent.com/76513466/137642169-3ca32034-03b7-4c89-af33-9693700a87c3.png)

## Feature Selection and Selection
There are various methods of feature selection and extraction but for the sake of this research we shall be discussing about an example of each of them applied in this 
research.
- TF-IDF Vectorizer: The term "term frequency" (TF) refers to the frequency of occurrence of words across varied documents, while the term "inverse document frequency" refers to the frequency of the phrase as it occurs in various documents. As opposed to TF, IDF stands for Inverse Document Frequency, which assesses a word's significance based on the frequency it appears in different documents. The common word theory suggests that common words that appear more frequently are judged less meaningful. TF-IDF is based on this principle. TF-IDF is a tool that converts text into vectors. The TfidfVectorizer analyzes the collection of documents before producing a TF-IDF matrix in numerical form.

- Word Stemming: The technique of stemming is used to extract the base form of words by removing their affixes. In stemming, we get the key word/root for various words. Example: legal, illegal, legality, legislation. Here the root word is 'legal'. The Portstemmer library tool is a python tool used to perform word stemming on text data.

- Word cloud: A word cloud is a textual data visualisation that allows anyone to see the terms with the highest frequency inside a given body of text in a single glance. Typically, word clouds are used to process, analyse, and disseminate qualitative sentiment data. Example of a word cloud visualization is the figure below that displays the most frequent words in the theme column of the annotated dataset employed for this 
research project.

![image](https://user-images.githubusercontent.com/76513466/137644725-1e0ff0ff-ec98-4f9e-9a43-3b149e7afa23.png)

## Analysis and Design
Initially, the selected news article dataset retrieved from Kaggle for the detection of fake news undergo the four data preprocessing phases including data acquisition, data cleaning, data reduction, data transformation and data integration phases. Upon the integration of the annotated dataset with python libraries and dependencies listed above for data wrangling. The initial step involves identifying and replacing the missing rows from the integrated dataset regarded as noisy data with empty strings to avoid deformation of analysis after which the title column and author columns were concatenated to ease data modelling and renamed as theme column. The second stage requires separating the theme column and news article labels into two separate dataframes. Next is to perform data reduction by feature extraction through the process of word stemming in extracting the root words of the word texts in the theme dataframe utilizing the PortStemmer library. Also, Stopwords which are trivial words with little or no importance to the aim of this research analysis were removed to not only ensure but also improve performance of the binary classifier. The root words obtained from the feature extraction process were visualized with word cloud data visualization plot to show the most frequent words and the top 10 most reoccurring words in the theme dataframe. Data transformation is achieved with the TF-IDF vectorizer; a feature selector employed to convert the word texts into a value-by-frequency matrix based on the repetition of different word. The theme dataframe is indicated as the predictor or independent variable and the label dataframe as the dependent variable before integration into the classification algorithm.

## Evaluation Metrics
- Accuracy score: The total number of right predictions of unreliable (1) and reliable (0) news articles made by the fitted regression model is the model's performance accuracy score. The results reveal what proportion of the fitted regression model's predictions were correct. The overall error rate is calculated by subtracting 1 from the accuracy score.
![image](https://user-images.githubusercontent.com/76513466/137643762-1d7e8eda-fe55-47e4-9080-27189904bd23.png)

- Confusion Matrix: This classification accuracy measure visualises the correlation coefficients of the predicted and real values and computes the True Positive, True Negative, False Positives, and False Negatives classes of news postings. The resulting confusion matrix shows that 3101 news articles were correctly identified (True Positive), 137 were wrongly identified (False Positive), 2979 were correctly rejected (True Negative), and 23 were incorrectly rejected (False Negative).

![image](https://user-images.githubusercontent.com/76513466/137643784-769637dc-4c7b-46f8-b989-ca3ec7065ca5.png)

- Sensitivity: The true positive rate, or sensitivity represents the number of legitimately positive but untrustworthy news stories that have been appropriately identified. The acquired score represents the percentage of news articles successfully recognized as untrustworthy, which is 99.26%.

![image](https://user-images.githubusercontent.com/76513466/137643808-5e0b2294-dc8b-494d-a680-3a7a2bb300c2.png)

- Specificity: This calculates the percentage of carefully selected trustworthy news articles. The percentage of news articles that are accurately recognized as dependable or true news. The false positive rate, or 1 — specificity.

![image](https://user-images.githubusercontent.com/76513466/137643839-54f3f533-b524-465c-a69f-709c38e494aa.png)

- F1-Score: The harmonic mean of the precision and recall accuracy scores is the F1-score. The harmonic mean of precision and recall is used to get the standard F1-score. The F-score is a popular metric for assessing information retrieval systems like search engines, as well as a variety of machine learning models, particularly in natural language processing. The fitted logistic regression model has an F1-score of 97.5 percent.

![image](https://user-images.githubusercontent.com/76513466/137643858-7aa7de12-3219-4105-a060-31b1369e466e.png)

![image](https://user-images.githubusercontent.com/76513466/137644232-5f66e4d8-63b2-4bad-a7c9-2a7d91fbcb39.png)

- Receiver Operating Characteristics (ROC) Curve: This is a graph plot that shows how well a binary classification model performs. As shown below, the ROC curve depicts the performance of this binary classification.  The ROC curve is produced by calculating and plotting the true positive rate against the false positive rate for a single classifier at a variety of thresholds. 

![image](https://user-images.githubusercontent.com/76513466/137644259-ea38665c-5c86-444b-a82d-3357f408ff14.png)

- Receiver Operating Characteristics (ROC) AUC Score: This is used to determine the ROC curve's threshold. When the AUC score approaches 0.5, it suggests that the binary classifier is doing poorly, and when it is close to 1, it shows that the binary classification model is performing well. 
The resulting ROC AUC score reveals the model accuracy threshold, 0.974336 close to 1, indicating the fitted classifier's excellent efficacy.

![image](https://user-images.githubusercontent.com/76513466/137644277-11844e1c-ba0c-497b-8a8e-3e0ca90fef04.png)

## Result and Conclusion
The integration of a news article dataset into a machine learning logistic regression model evaluated at 98.51 percent accuracy yielded successful results from the binary classification of false news or unreliable news articles and reliable posts. The ROC AUC of 97.4% indicates the binary classifier's accuracy at various thresholds. This result demonstrates the high accuracy of the logistic regression model for binary classification of news items utilizing the collected news features and a train-test cross validation technique. To distinguish between true and fake news, logistic regression employed its sigmoid function to classify binary inputs and a support vector machine to classify social media postings and news articles.
These machine learning algorithms should be employed to further research on fake news detection. Another limitation of this research is that it fails to consider other forms of news such as satire news which may have been included in the news article dataset. Satire news is a literary work consisting of fabrications or fictional play with the aim of entertaining its audience. Another form of news that was failed to consider are neutral opinions which only make statements without any form of motive behind it. For example: “USA represents the United States of America”. Another form of words that were not considered are slangs. 
Another possible area of research is tuning the hyperparameters of the model to enhance the efficacy of the adopted model. Also, introducing auxiliary information such as news styling and number of readers or user engagement with the reported news can also play an important role in determining whether the published news article or online news report is real or fake.



## Short Summary

Evaluating the accuracy of Fake News detection with Logistic Regression Model - Link 
•	Utilised word clouds to demonstrate most frequent words in political fake news articles in python.
•	Executed feature extraction and reduction with Portstemmer and TF-IDF function to optimise dataset by 60%
•	Built a logistic regression model to classify 20800 political news articles with NLP and Scikit-learn libraries. 








