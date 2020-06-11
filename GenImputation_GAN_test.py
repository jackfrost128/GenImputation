import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D, Dropout
from tensorflow.keras.regularizers import l1
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import load_model

input_name = 'data/yeast_genotype_test.txt'
df_ori = pd.read_csv(input_name, sep='\t', index_col=0)
df_ori.shape

# one hot encode
test_X = to_categorical(df_ori)
test_X.shape

# returns a compiled model
from tensorflow.keras.models import load_model
GAN = load_model('model/mygan.h5')

# hyperparameters
missing_perc = 0.1

test_X_missing = test_X.copy()
test_X_missing.shape

def cal_prob(predict_missing_onehot):
    # calcaulate the probility of genotype 0, 1, 2
    predict_prob = predict_missing_onehot[:,:,1:3] / predict_missing_onehot[:,:,1:3].sum(axis=2, keepdims=True)
    return predict_prob[0]


avg_accuracy = []
for i in range(test_X_missing.shape[0]):
    # Generates missing genotypes
    missing_size = int(missing_perc * test_X_missing.shape[1])
    missing_index = np.random.randint(test_X_missing.shape[1],
                                      size=missing_size)
    test_X_missing[i, missing_index, :] = [1, 0, 0]

    # predict
    predict_onehot = GAN.predict(test_X_missing[i:i + 1, :, :])
    # only care the missing position
    predict_missing_onehot = predict_onehot[0:1, missing_index, :]

    # calculate probability and save file.
    predict_prob = cal_prob(predict_missing_onehot)
    #     pd.DataFrame(predict_prob).to_csv('results/{}.csv'.format(df_ori.index[i]),
    #                                       header=[1, 2],
    #                                       index=False)

    # predict label
    predict_missing = np.argmax(predict_missing_onehot, axis=2)
    # real label
    label_missing_onehot = test_X[i:i + 1, missing_index, :]
    label_missing = np.argmax(label_missing_onehot, axis=2)
    # accuracy
    correct_prediction = np.equal(predict_missing, label_missing)
    accuracy = np.mean(correct_prediction)
    print('{}/{}, sample ID: {}, accuracy: {:.4f}'.format(
        i, test_X_missing.shape[0], df_ori.index[i], accuracy))

    avg_accuracy.append(accuracy)

print('The average imputation accuracy' \
      'on test data with {} missing genotypes is {:.4f}: '
    .format(missing_perc, np.mean(avg_accuracy)))
