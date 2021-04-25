import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # the goal is to predict to which class new feature will belong

    # KNN_Project_Data csv file into a dataframe
    df = pd.read_csv('KNN_Project_Data')

    # checking data information
    print(df.head())
    print(df.info())

    # pairplot with the hue indicated by the TARGET CLASS column
    sns.pairplot(df, hue = 'TARGET CLASS')

    # StandardScaler() object
    scaler = StandardScaler()

    # fitting scaler to the features
    scaler.fit(df.drop('TARGET CLASS', axis = 1))

    # transforming the features to a scaled versio
    scaled_feat = scaler.transform(df.drop('TARGET CLASS', axis = 1))

    # converting the scaled features to a dataframe
    df_feat = pd.DataFrame(scaled_feat, columns = df.columns[:-1])
    print(df_feat.head())

    # splitting the data into a training set and a testing set
    X = df_feat
    y = df['TARGET CLASS']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

    # KNN model instance with n_neighbors = 1
    knn = KNeighborsClassifier(n_neighbors = 1)

    # fitting KNN model to the training data
    knn.fit(X_train, y_train)

    # predicting the values using KNN model and X_test
    predictions = knn.predict(X_test)

    # confusion matrix and classification report
    print(confusion_matrix(y_test, predictions), '\n')
    print(classification_report(y_test, predictions))

    # for loop that trains various KNN models with different k values; keeping track of the error_rate for each of these
    # models with a list
    error_rate = []
    for i in range(1, 40):
        knn = KNeighborsClassifier(n_neighbors = i)
        knn.fit(X_train, y_train)
        predictions_i = knn.predict(X_test)
        error_rate.append(np.mean(predictions_i != y_test))

    # visualization of the information from for loop
    plt.figure(figsize = (10, 7))
    plt.plot(range(1, 40), error_rate, linestyle = 'dashed', color = 'blue', marker = 'o', markerfacecolor = 'red',
             markersize = 10)
    plt.title('Error rate vs. K value')
    plt.xlabel('K')
    plt.ylabel('Error Rate')

    # retraining the model with the best K value and re-doing the classification report and the confusion matrix
    knn = KNeighborsClassifier(n_neighbors = 31)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(confusion_matrix(y_test, predictions), '\n')
    print(classification_report(y_test, predictions))

    plt.show()


if __name__ == '__main__':
    main()
