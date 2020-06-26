1. This project is a machine learning model to identify if emails are spam or not. The dataset used is from the UCI Machine Learning Dataset.
https://archive.ics.uci.edu/ml/datasets/Spambase

2. Due to issues and constraints with Azure Machine Learning Studio, this was done locally with Jupyter Notebook. The dataset was downloaded and the names of each column was cleaned in a new file called 'modified.names'.
Pandas was used to create a dataframe for the dataset. Seaborn was used to graph the dataset, and observe for any null values.
sklearn was used to create machine learning models and since this is supervised learning, it used to fit/train a model on training data, then test the model on testing data.

3.

1. obtained spambase dataset from UCI Machine Learning Process
2. Cleaned dataset with Pandas and Seaborn
3. Used sklearn, and imported train_test_splot to split dataset into train and test. 
4. To select the optimal model, also imported different algorithms and fitted the training set into each model.
5. First model used to predict the testing set was fitting the training set with a logistic regression model then printed the confusion_matrix
6. Second model was the Gaussian Naive Bayes
7. third model was a weighted logistic regression model, as the dataset was does not have equal amount of spam vs. non-spam (39.4% is non-spam and 60.6% is spam)
8. The last model was the KNeighbors Classifier.

After comparing the precision and recall of each model's confusion matrix, the logistic regression model was shown to be the better model for this dataset.