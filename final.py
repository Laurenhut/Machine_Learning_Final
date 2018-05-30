
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# define baseline model
'''def baseline_model():
	# create model
	model = Sequential()

    model.add(Dense(8, input_dim=1023, activation='relu'))
    model.add(Dense(8, input_dim=1023, activation='relu'))
	model.add(Dense(37, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
'''

def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=1023, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(37, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataframe = pandas.read_csv("train.csv", delimiter = ',', header=None)
#dataset = dataframe.values
data = np.loadtxt('train.csv', delimiter = ',')
#X = dataset[:,0:4].astype(float)
#Y = dataset[:,4]
attributes = data[:,1:] #x
classes=data[:,0] #y
#print(x)
#print(" hi")
#print(x[0].shape)
#print("classes")
#print(classes.shape)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(classes)
encoded_Y = encoder.transform(classes)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=10, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)


results = cross_val_score(estimator, attributes, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
