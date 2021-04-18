# Classifying-income-data-using-Support-Vector-Machines
We will build a Support Vector Machine classifier to predict the income bracket of a given person based on 14 attributes. Our goal is to see where the income is higher or lower than $50,000 per year. Hence this is a binary classification problem. We will be using the census
income dataset available at https://archive.ics.uci.edu/ml/datasets/Census+Income.
One thing to note in this dataset is that each datapoint is a mixture of words and numbers.
We cannot use the data in its raw format, because the algorithms don't know how to deal
with words. We cannot convert everything using label encoder because numerical data is
valuable. Hence we need to use a combination of label encoders and raw numerical data to
build an effective classifier.

Create a new Python file and import the following packages:
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn import cross_validation
```

We will be using the file income_data.txt to load the data. This file contains the income
details:
```
# Input file containing data
input_file = 'income_data.txt'
```

In order to load the data from the file, we need to preprocess it so that we can prepare it for
classification. We will use at most 25,000 data points for each class:
```
# Read the data
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000
```

Open the file and start reading the lines:
```
with open(input_file, 'r') as f:
 for line in f.readlines():
  if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
   break
  if '?' in line:
   continue
``` 

Each line is comma separated, so we need to split it accordingly. The last element in each
line represents the label. Depending on that label, we will assign it to a class:
```
 data = line[:-1].split(', ')
 if data[-1] == '<=50K' and count_class1 < max_datapoints:
   X.append(data)
   count_class1 += 1
 if data[-1] == '>50K' and count_class2 < max_datapoints:
   X.append(data)
   count_class2 += 1
 ```
 
Convert the list into a numpy array so that we can give it as an input to the sklearn
function:
```
# Convert to numpy array
X = np.array(X)
```

If any attribute is a string, then we need to encode it. If it is a number, we can keep it as it is.
Note that we will end up with multiple label encoders and we need to keep track of all of
them:
```
# Convert string data to numerical data
label_encoder = []
X_encoded = np.empty(X.shape)
for i,item in enumerate(X[0]):
 if item.isdigit():
    X_encoded[:, i] = X[:, i]
 else:
    label_encoder.append(preprocessing.LabelEncoder())
    X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)
```

Create the SVM classifier with a linear kernel:
```
# Create SVM classifier
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
```

Train the classifier:
```
# Train the classifier
classifier.fit(X, y)
```

Perform cross validation using an 80/20 split for training and testing, and then predict the
output for training data:
```
# Cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=5)
classifier = OneVsOneClassifier(LinearSVC(random_state=0))
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)
```

Compute the F1 score for the classifier:
```
# Compute the F1 score of the SVM classifier
f1 = cross_validation.cross_val_score(classifier, X, y, scoring='f1_weighted', cv=3)
print("F1 score: " + str(round(100*f1.mean(), 2)) + "%")
```

Now that the classifier is ready, let's see how to take a random input data point and predict
the output. Let's define one such data point:
```
# Predict output for a test datapoint
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 
'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']
```

Before we can perform prediction, we need to encode this data point using the label
encoders we created earlier:
```
# Encode test datapoint
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
 if item.isdigit():
  input_data_encoded[i] = int(input_data[i])
 else:
  input_data_encoded[i] = int(label_encoder[count].transform(input_data[i]))
  count += 1
input_data_encoded = np.array(input_data_encoded)
```

We are now ready to predict the output using the classifier:
```
# Run classifier on encoded datapoint and print output
predicted_class = classifier.predict(input_data_encoded)
print(label_encoder[-1].inverse_transform(predicted_class)[0])
```

If you run the code, it will take a few seconds to train the classifier. Once it's done, you will
see the following printed on your Terminal:
```
 F1 score: 66.82%
``` 
 
You will also see the output for the test data point:
```
 <=50K
``` 
 
If you check the values in that data point, you will see that it closely corresponds to the data
points in the less than 50K class. You can change the performance of the classifier (F1 score,
precision, or recall) by using various different kernels and trying out multiple combinations
of the parameters.

 
