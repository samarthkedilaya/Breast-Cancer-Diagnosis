# BREAST CANCER DIAGNOSIS USING MACHINE LEARNING



## Description:
[Breast cancer](https://www.cancer.org/cancer/breast-cancer.html) starts when cells in the breast begin to grow out of control. These cells usually form a tumor that can often be seen on an x-ray or felt as a lump.

There are two main kinds of tumors:

1. Benign - tumors are non-cancerous

2. Malignant - tumors are cancerous.



## Goals:
To classify whether the breast cancer is benign or malignant.

Comparing the accuracy of two supervised and one unsupervised machine learning algorithms.



## Dataset:
The [dataset](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/data/data.csv) contains 569 instances, 357 are labeled as B (benign) and 212 as M (malignant).

30 features that are computed here come from a digitized image of a fine needle aspirate (FNA) of breast mass and they describe characteristics of the cell nuclei present in the image.



## Supervised ML Algorithms
### [Multi-Layer Perceptron](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/Supervised.py)
1. Data Preprocessing - Split the data into train and test set (80:20). Then apply Minmax normalization which scales all the feature values to lie in the range of 0 to 1. 

![mlp](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/images/MLP.png)

2. Forward propagation -  [Sigmoid Activation Function](https://en.wikipedia.org/wiki/Sigmoid_function)

3. [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)


#### MLP Results: 
Confusion Matrix (Train data) -
[[283   2]
[9 161]]

Confusion Matrix (Test data) -
[[72  0]
[10 32]]

Final accuracy of train data at 50th epoch is 97.58 %

Final accuracy of test data at 50th epoch is 91.23 %


### [Logistic Regression](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/Supervised.py)
1. Data Preprocessing - Split the data into train and test set in (80:20) proportion and apply standardization, which centers the values of each feature column, setting it to have a mean of 0 and a standard deviation of 1. [new_x = (x - mean)/std]
2. Forward Propagation
3. Cost function or Cross Entropy or Likelihood function

![CF](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/images/CF.png)

4. [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)

![GD1](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/images/GD1.png)

![GD2](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/images/GD2.png)

5. [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation)

![BP1](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/images/BP1.png)

![BP2](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/images/BP2.png)

6. Classification


#### LR Results:
Confusion Matrix (Train data) - 
[[265   4]
 [5 181]]
 
Confusion Matrix (Test data) - 
[[62 10]
 [ 0 72]]

Final accuracy of Train data at 3000th epoch is 98.02 %

Final accuracy of Test data at 3000th epoch is 93.05 %



## Unsupervised ML Algorithm
### [K-Means Clustering](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/Unsupervised.py)
![KM](https://github.com/samarthkedilaya/Breast-Cancer-Diagnosis/blob/master/images/KM.png)

1. Mapped M = malignant or B = benign to 1 or 0 respectively and preprocessed the data by min max normalization

2. Created train and test data by splitting 

3. Initialized the KMeans cluster module.

<p align="center"><strong>
  Clusters = KMeans(n_clusters=2, n_init=10,max_iterations=300)
  </strong>
  </p>
  <p align="center"><sub>
  where n_init = No. of times k-means will run with different centroid seeds<br>
  max_iterations = Max. no. of iterations of k-means for a single run
  </sub></p>


#### KM Results:
Dataset divided as 80% train dataset & 20% test dataset
[[80  0]
 [12 22]]
 
Final Accuracy is 89.47%


