import pandas as pd
import numpy as np
from dataset import DataSet
from model import Ensemble

def multiple_models(data_hhb, data_hbo2, data_na, data_ca, Ytrain, Xtest):
    print('Fit the models for ensemable')
    model1 = Ensemble()
    model2 = Ensemble()
    model3 = Ensemble()
    model4 = Ensemble()
    model1.fit(data_hhb, Ytrain.hhb)
    model2.fit(data_hbo2, Ytrain.hbo2)
    model3.fit(data_na, Ytrain.na)
    model4.fit(data_ca, Ytrain.ca)

    print('Predict the target values')
    tdata = DataSet(Xtest)
    tdata_hhb, tdata_hbo2, tdata_na, tdata_ca, _ = tdata.data_proccessing(False)
    hhb = model1.predict(tdata_hhb)
    hbo2 = model2.predict(tdata_hbo2)
    na = model3.predict(tdata_na)
    ca = model4.predict(tdata_ca)
    na = np.exp(na - 1.5) - 2

    return np.concatenate([hhb.reshape((-1, 1)), hbo2.reshape((-1, 1)), ca.reshape((-1, 1)), na.reshape((-1, 1))], axis=1)

if __name__ == "__main__":

    train = pd.read_csv('./data/train.csv', index_col='id')
    Xtest = pd.read_csv('./data/test.csv', index_col='id')
    submission = pd.read_csv('./data/sample_submission.csv', index_col='id')

    cols = ['hhb', 'hbo2', 'ca', 'na']
    Ytrain = train[cols]
    Ytrain.na = np.log(Ytrain.na + 2) + 1.5
    Xtrain = train.drop(cols, axis=1)

    print(train.shape, Xtest.shape, submission.shape)

    data = DataSet(Xtrain, Ytrain)
    data_hhb, data_hbo2, data_na, data_ca, Ytrain = data.data_proccessing(True)

    preds = multiple_models(data_hhb, data_hbo2, data_na, data_ca, Ytrain, Xtest)
    preds = pd.DataFrame(data=preds, columns=cols, index=submission.index)
    preds.to_csv('./submission_final.csv')
    print("Completed")