import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

n_clusters = 2


# n_init=20
# max_iter=300

# reading the data
def readDataSet():
    return pd.read_csv('data/data.csv')


# preprocessing the data
def cleanseDataSet(bcData):
    df_x = bcData.loc[:, "radius_mean":"fractal_dimension_worst"]  # 30 features
    df_x = df_x.astype(np.float32)
    df_y = bcData.loc[:, "diagnosis"]
    df_y = df_y.replace("M", 1)
    df_y = df_y.replace("B", 0)
    return df_x, df_y


# Normalized Data
def normalizeData(x):
    for idx in ("radius_mean", "fractal_dimension_worst"):
        x[idx] = x[idx] - min(x[idx]) / (max(x[idx]) - min(x[idx]))
    return x


def createTrainTestDataSet(df_x, df_y):
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=.2, random_state=4)
    return x_train, x_test, y_train, y_test


def kmeansClustering(x_train, y_train):
    model = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300)
    fittedModel = model.fit(x_test, y_test)
    return fittedModel


def envSetup():
    # print("Read data from CSV ::")
    bcData = readDataSet()
    df_x, df_y = cleanseDataSet(bcData)
    df_x = normalizeData(df_x)
    # "Dataset Normalized successfully!!!"#"Dataset reduced to 9 columns"
    x_train, x_test, y_train, y_test = createTrainTestDataSet(df_x, df_y)
    print("Dataset divided as 80% train dataset & 20% test dataset")
    return x_train, x_test, y_train, y_test


def choiceKMeans():
    fittedModel = kmeansClustering(x_train, y_train)
    predictions = getPrediction(fittedModel, x_test)
    confusion_mat = getConfusionMatrix(y_test, predictions)
    print(confusion_mat)
    print("Get the Accuracy ::")
    accuracy = getAccuracy(y_test, predictions) * 100
    print(accuracy)


def getPrediction(fittedModel, x_test):
    predictions = fittedModel.predict(x_test)
    return predictions


def getConfusionMatrix(y_test, predictions):
    return confusion_matrix(y_test, predictions)


def getAccuracy(y_test, predictions):
    return accuracy_score(y_test, predictions)


####main

x_train, x_test, y_train, y_test = envSetup()
choiceKMeans();
