# Importing the libraries
import pandas as pd
import pickle

dataset = pd.read_csv('TH.csv')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.svm import SVC
regressor = SVC(kernel='linear')

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))