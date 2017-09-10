import pandas as pd
import numpy as np
import time
import pickle
import sys
import os
from datetime import timedelta
from array import array
from azureml.logging import get_azureml_logger
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
except ImportError:
    import pip
    package_name='matplotlib'
    pip.main(['install', package_name])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
try:
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.tsa.arima_model import ARIMA
except ImportError:
    package_name='statsmodels'
    pip.main(['install', package_name])
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.tsa.arima_model import ARIMA
def TrainTimeSeries(dataset, p, d, q, freq):
    if freq > 0:
        model = ARIMA(dataset,order= (p,d,q))
        return model

def parser(x):
    d = timedelta(hours=x[0])
    dt = pd.datetime(2017, 6, 19, 0, 0, 0, 0) + d
    return dt

def PrepareTrainData(dataset):
    dataset2 = dataset[['time']]
    dataset3 = dataset2.astype(np.float64)
    d2 = pd.DataFrame(np.zeros(shape=(len(dataset3), 1)))
    len(dataset3)
    for idx in range(len(dataset3)):
        d = timedelta(hours=dataset3.iloc[idx, 0])
        d2.iloc[idx, 0] = pd.datetime(2017, 6, 19, 0, 0, 0, 0) + d

    #print(dataset.info())
    d3 = dataset[['N1725']]
    frames = [d2, d3]
    d4 = pd.concat(frames, axis=1)
    #d5 = d4
    #d5.index = (d4[[0]])[0]
    # d5.index
    #d6 = d5['N1725'].astype(np.float64)
    d5 = pd.DataFrame({'N1725':d4.iloc[0:126,1].astype(np.float64)}) #d4.iloc[0:108,0:1]
    d5.index = (d4.iloc[0:126,0])
    d5.index.name = 'time'
    return d5

def PrepareFcstData(strt, stp):
    df = pd.DataFrame(np.zeros(shape=(stp-strt+1, 1)))
    fcst = pd.DataFrame(np.zeros(shape=(stp-strt+1, 1)))

    for idx in range(strt, strt + len(df)):
        d = timedelta(hours=idx)
        df.iloc[idx - strt, 0] = pd.datetime(2017, 6, 19, 0, 0, 0, 0) + d
    # print(df)
    fcst.index = df.iloc[:, 0]
    fcst.index.name = 'time'
    fcst.columns = ['forecast']
    return fcst

def CalResultsArr(model):
    results_AR = model.fit(disp=-1)
    #results_AR = model.fit(method = 'css-mle', disp = 0)
    #print(results_AR.summary())
    plt.plot(results_AR.fittedvalues, color='red')
    return results_AR

def PredictFcst(res_AR, strt, stp):
    fst = PrepareFcstData(strt,stp)


    fst['forecast']=res_AR.predict(start = strt-1, end= stp, dynamic= True)
    return fst

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



#print("Pandas Version:"+pd.__version__)
# initialize the logger
run_logger = get_azureml_logger() 

# create the outputs folder
os.makedirs('./outputs', exist_ok=True)

# read and parse data
dataset1 = pd.read_csv("TimeSeriesDataset.csv")
dataset2 = PrepareTrainData(dataset1)

# initialize plots
plt.close()
#print(dataset2)
fig = plt.figure()
p = plt.plot(dataset2, color='blue',label='Original')

# train model
ar_mdl = TrainTimeSeries(dataset2, 6, 0, 2, 12)
ar_res = CalResultsArr(ar_mdl)

# predict future values
f = PredictFcst(ar_res, 127, 151)

# create and save plots
plt.plot(f,color='green',label='Forecast')
#plt.show(block=False)
fig.savefig("./outputs/forecast.png")

# serialize the model on disk in the special 'outputs' folder
fldr = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + "outputs/"
os.makedirs(fldr, exist_ok=True)
print ("Export the model to model.pkl")
fl = open(fldr+"model.pkl", 'wb')
pickle.dump(ar_res, fl)
fl.close()

print ("Export the model to model.pkl in outputs directory")
fl = open("./outputs/model.pkl", 'wb')
pickle.dump(ar_res, fl)
fl.close()

# calculate rmse and other metrics
trgt = dataset2['N1725'].as_matrix()
pred = ar_res.fittedvalues.as_matrix()
rmseval = rmse(pred,trgt)
print("***************")
run_logger.log("RMSE",rmseval)
print("RMSE:    ", rmseval)
print("***************")
print("Forecast Values:")
print(f)
print("***************")
print("AIC:     ",ar_res.aic)
print("BIC:     ",ar_res.bic)
print("HQIC:    ", ar_res.hqic)
print("***************")
print(ar_res.summary())
