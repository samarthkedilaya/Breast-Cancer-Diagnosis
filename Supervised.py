import numpy as np
from scipy.special import expit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


#constant bias input
biasInput=1
#eta
learningrate=0.05
#momentum
alpha=0
#number of hidden units in hidden layer
hidden_units=15
#total number of epochs
epochs=3000

#input layer weights: wij
input_weight = np.random.uniform(-0.05,0.05,(30,hidden_units))
#output layer weights: wjk
hidden_weight = np.random.uniform(-0.05,0.05,(hidden_units,2))

#input layer bias weights
inputlayer_bias_weights = np.random.uniform(-0.05,0.05,(1,hidden_units))
#hidden layer bias weights applied to 2 neurons in output layer
hiddenlayer_bias_weights = np.random.uniform(-0.05,0.05,(1,2))

#previous delta input bias weight
delta_input_bias_weight = np.zeros((1,hidden_units))
#previous delta hidden bias weight
delta_hidden_bias_weight = np.zeros((1,2))

#previous delta input  weight
delta_input_weight = np.zeros((30,hidden_units))
#previous delta hidden weight
delta_hidden_weight = np.zeros((hidden_units,2))

#lists to store the train and test data accuracy of each epoch
accuracy_train=[]
accuracy_test=[]

#max-min normalization
def read_preprocess():
    #read the data from csv file
    data = pd.read_csv('data.csv')
    #convert malignant to 1 and benign to 0
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    #extract 30 features
    tempData = data[data.columns[2:32]]
    labels = np.matrix(data['diagnosis'].as_matrix()).transpose()
    input_data = tempData.values
    #separate train and test data and labels. Train Data:Test Data =80:20
    t_data, testing_data, train_label, test_label = train_test_split(input_data, labels, test_size=0.2)
    #train data preprocess using min-max normalization
    minv = np.amin(t_data, axis=0)
    maxv = np.amax(t_data, axis=0)
    train_data = (t_data - minv) / (maxv - minv)
    #test data preprocess using min-max normalization
    minva = np.amin(testing_data, axis=0)
    maxva = np.amax(testing_data, axis=0)
    test_data = (testing_data - minva) / (maxva - minva)
    return train_data,test_data,train_label,test_label


#function to calculate sigma(wx)=1/(1+e^-wx)
def sigmoid_function(wx):
    return expit(wx)

#initiate forward propagation and call in backpropagation when epoch is not 0 and data is Training data
def forwardpropagation(data,label,data_name,epoch):
    global input_weight,hidden_weight,biasInput,inputlayer_bias_weights,hiddenlayer_bias_weights,delta_input_bias_weight,delta_input_weight,delta_hidden_bias_weight,delta_hidden_weight
    predict_data=[]
    label_cfm=[]
    correct=0 #counter to keep track of correct prediction in the dataset for a epoch
    for i in range(data.shape[0]): # iterate through the entire dataset
        inputData = np.reshape(data[i],(1,data[i].shape[0]))
        wx_input=np.dot(inputData,input_weight) + (biasInput*inputlayer_bias_weights)  #calculate summation of wxij
        sigma_hidden = sigmoid_function(wx_input) #calculate hj=sigma(wxij)
        wx_hidden = np.dot(sigma_hidden,hidden_weight) + (biasInput*hiddenlayer_bias_weights) #calculate summation of wjk
        sigma_output = sigmoid_function(wx_hidden) #calculate ok=sigma(wjk)
        prediction = np.argmax(sigma_output)  #get the index of the neurorn with the hightest sigma(output)
        if(epoch==epochs-1):  #if epoch is 49 then store the prediction of each dataset and label corresponding to it
            predict_data.append(prediction)
            label_cfm.append(label[i,0])
        if(prediction==label[i,0]):  #if prediction matches the label increment the correct prediction counter
            correct=correct+1
        if(data_name == 'Training' and epoch != 0): # Initiate backpropagation if it is Training dataset and epoch is not 0
            delta_input_bias_weight,delta_input_weight,delta_hidden_bias_weight,delta_hidden_weight = backpropagation(inputData,sigma_hidden,sigma_output,label[i,0])
    if(epoch==epochs-1): # Create confusion matrix for the 50th epoch
        cfm=confusion_matrix(label_cfm,predict_data)
        print(data_name," Confusion Matrix of MLP ")
        print(cfm)
    return (correct/len(data))*100 #calculate the percentage of accuracy of one epoch of the entire dataset

#Calculate the hidden and output error and initiate weight update
def backpropagation(data,sigma_hidden,sigma_output,label):
    global biasInput,hidden_weight,hiddenlayer_bias_weights,learningrate,alpha,delta_hidden_bias_weight,delta_hidden_weight
    target = np.insert((np.zeros((1,1))+0.1),label.astype(int),0.9)
    output_error = sigma_output * (1 - sigma_output) * (target - sigma_output) #calculate output_error=ok*(1-ok)*(t-ok)
    hidden_error = sigma_hidden * (1 - sigma_hidden) * np.dot(output_error,np.transpose(hidden_weight)) #calculate hidden_error=hj*(1-hj)*(output_error*wjk)
    delta_output = weightupdate_hiddenlayer(sigma_hidden,output_error) #update output layer weight
    delta_hidden = weightupdate_inputlayer(data,hidden_error) #update hidden layer weight
    return delta_hidden[0],delta_hidden[1],delta_output[0],delta_output[1]

#calculate wkj=wkj+n*output_error*hj
def weightupdate_hiddenlayer(sigma_hidden,output_error):
    global biasInput,hiddenlayer_bias_weights,hidden_weight,learningrate,alpha,delta_hidden_weight,delta_hidden_bias_weight #calculating n*output_error*hj of bias weight
    delta_bias = (learningrate * np.dot(output_error,biasInput)) + (alpha * delta_hidden_bias_weight)  #wij=wij+n*output_error*hj of bias weight
    hiddenlayer_bias_weights = hiddenlayer_bias_weights + delta_bias #calculate n*output_error*hj of input weight
    delta_weight = (learningrate * np.dot(np.transpose(sigma_hidden),output_error)) + (alpha * delta_hidden_weight)
    hidden_weight = hidden_weight + delta_weight #wij=wij+n*output_error*hj of hidden weight
    return delta_bias,delta_weight

#calculate wij=wij+n*hidden_error*xi
def weightupdate_inputlayer(data,hidden_error):
    global biasInput,inputlayer_bias_weights,input_weight,learningrate,alpha,delta_input_bias_weight,delta_input_weight
    delta_bias = (learningrate * np.dot(hidden_error,biasInput)) + (alpha * delta_input_bias_weight)  #calculate n*hidden_error*xi of bias weight
    inputlayer_bias_weights = inputlayer_bias_weights + delta_bias #calculate wij=wij+n*hidden_error*xi of bias weight
    delta_weight = (learningrate * np.dot(np.transpose(data),hidden_error)) + (alpha * delta_input_weight) #calculate n*hidden_error*xi of input layer weight
    input_weight = input_weight + delta_weight #calculate wij=wij+n*hidden_error*xi of input layer weight
    return delta_bias,delta_weight

#main function
def MLP():
    #read the data from csv file and preprocess the data
    train_data, test_data, train_label, test_label = read_preprocess()
    for eachepoch in range(epochs): #iterate over 50 epochs
        train_acc=forwardpropagation(train_data,train_label,'Training',eachepoch) #initiate forward propagation of Training data
        accuracy_train.append(train_acc) #add accuracy to list
        test_acc=forwardpropagation(test_data,test_label,'Testing',eachepoch) #initiate forward propagation of test data
        accuracy_test.append(test_acc) #add accuracy to list
    return accuracy_test
    # print("Accuracy of Train data : ",accuracy_train)
    # print("Accuracy of Test data : ",accuracy_test)
    # #generate plot of epoch vs accuarcy
    # plt.plot(accuracy_train,color='green',label='Train Data Accuracy')
    # plt.plot(accuracy_test,color='red',label='Test Data Accuracy')
    # plt.ylabel("Accuracy in %")
    # plt.xlabel("Epoch")
    # plt.legend(loc='best')
    # plt.show()



###################################Logistic Regression######################################
data = pd.read_csv('data.csv')
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
tempData = data[data.columns[2:32]]
labels = data['diagnosis']
labels = labels.values.reshape(labels.shape[0], 1)
trainData = tempData.loc[0:454, tempData.columns[0:]]
trainLabels = labels[0:455]
testData = tempData.loc[0:143, tempData.columns[0:]]
testLabels = labels[0:144]
numOfEpochs = 3000
learningRate = 0.01

mean = trainData.mean()
standardDeviation = trainData.std()
trainData = (trainData - mean)/standardDeviation

mean = testData.mean()
standardDeviation = testData.std()
testData = (testData - mean)/standardDeviation

def sigmoidActivation(x):
    return 1/(1+np.exp(-x))

def weightsInitilization(dataSize):
    global w, b
    w = np.zeros((dataSize, 1))
    b = 1
    return w, b

w, b = weightsInitilization(trainData.shape[1])


def propagation(weights, bias):
    size = trainData.shape[0]
    #Maximum likelihood estimate
    Z = np.dot(trainData, weights) + bias
    activation = sigmoidActivation(Z)
    cost = -np.sum(trainLabels*np.log(activation) - (1 - trainLabels)*np.log(1-activation))/size
    #Gradient ascent
    deltaZ = activation - trainLabels
    deltaW = np.dot(np.transpose(trainData), deltaZ)/size
    deltaB = np.sum(deltaZ)/size
    return deltaW, deltaB, cost

def optimization(weights ,bias):
    global numOfEpochs, learningRate
    costs = []
    trainAccuracies = []
    testAccuracies = []
    for i in range(numOfEpochs):
        deltaW, deltaB, cost = propagation(weights, bias)
        weights = weights - learningRate * deltaW
        bias = bias - learningRate * deltaB
        if i % 100 == 0:
            costs.append(cost)
        # Accuracy for training set:
        temp = findPrediction(weights, bias, trainData)
        trainPrediction = np.array(list(map(classify, temp)))
        trainPrediction = trainPrediction.reshape((trainPrediction.shape[0], 1))
        trainAccuracy = 100 - np.mean(np.abs(trainPrediction - trainLabels)) * 100
        trainAccuracies.append(trainAccuracy)
        # Accuracy for test set:
        temp = findPrediction(weights, bias, testData)
        testPrediction = np.array(list(map(classify, temp)))
        testPrediction = testPrediction.reshape((testPrediction.shape[0], 1))
        testAccuracy = 100 - np.mean(np.abs(testPrediction - testLabels)) * 100
        testAccuracies.append(testAccuracy)
        if (i == numOfEpochs - 1):
            cfm_train = confusion_matrix(trainLabels, trainPrediction)
            print("Training Confusion Matrix: ")
            print(cfm_train)
            print("Testing Confusion Matrix: ")
            cfm_test=confusion_matrix(testPrediction,testLabels)
            print(cfm_test)
    print("All train accuracies: ", trainAccuracies)
    print("All test accuracies: ", testAccuracies)
    print("The train accuracy is: ", trainAccuracy)
    print("The test accuracy is: ", testAccuracy)
    # costs = np.squeeze(costs)
    # plt.plot(costs)
    # plt.xlabel('No. of iteration')
    # plt.ylabel('Cost')
    plt.show()
    return trainAccuracies, testAccuracies, costs

def findPrediction(weights, bias, data):
    activation = sigmoidActivation(np.dot(data, weights) + bias)
    return activation

def classify(datum):
    if datum >= 0.5:
        return 1
    elif datum < 0.5:
        return 0

def logisticRegression():
    global w, b
    trainAccuracies, testAccuracies, costs = optimization(w, b)
    return  testAccuracies
    # plt.plot(trainAccuracies, '-b', label='train')
    # plt.plot(testAccuracies, '-r', label='test')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Logistic Regression with Learning Rate = 0.01 & Epochs = 3000')
    # plt.legend(loc='lower right')
    # plt.show()

if __name__ == '__main__':
    MLP_accuracy=MLP()
    LogisticRegression_accuracy=logisticRegression()
    plt.plot(MLP_accuracy, '-b', label='MLP test accuracy')
    plt.plot(LogisticRegression_accuracy, '-r', label='Logistic Regression test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

