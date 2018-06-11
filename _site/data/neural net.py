
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score,learning_curve
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# define baseline model
'''

def baseline_model(x,y):
    model = Sequential()
    model.add(Dense(10, input_dim=1023, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='relu'))

    model.add(Dense(38, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    history=model.fit(x, y, validation_split=0.10, epochs=200, batch_size=1, verbose=1)

    #print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

dataframe = pandas.read_csv("train.csv", delimiter = ',', header=None)
#dataset = dataframe.values
data = np.loadtxt("train.csv", delimiter = ',')

attributes = data[:,1:] #x
classes=data[:,0] #y
new_y = np_utils.to_categorical(classes)


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(classes)
encoded_Y = encoder.transform(classes)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(np.array(dummy_y).shape)
print(np.asarray(new_y).shape)
baseline_model(attributes,new_y)
#mod=baseline_model
#history = mod.fit(X, Y, validation_split=0.10, epochs=50, batch_size=30, verbose=0)
# list all data in history
#estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=10, verbose=0)
#kfold = KFold(n_splits=7, shuffle=True, random_state=seed)


#results = cross_val_score(estimator, attributes, dummy_y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(results.history.keys())

'''

def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=1023, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='relu'))
    model.add(Dropout(0.1))
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

attributes = data[:,1:] #x
classes=data[:,0] #y



# encode class values as integers
encoder = LabelEncoder()
encoder.fit(classes)
encoded_Y = encoder.transform(classes)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

estimator = KerasClassifier(build_fn=baseline_model, epochs=500, batch_size=10, verbose=1)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

#generates image
plt.figure()
plt.title("Neural Net")
plt.xlabel("Training examples")
plt.ylabel("Score")
train_sizes, train_scores, test_scores = learning_curve(estimator, attributes, dummy_y,  cv=kfold, verbose=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
         label="Cross-validation score")

plt.legend(loc="best")
plt.show()

#results = cross_val_score(estimator, attributes,dummy_y, cv=kfold)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
