import pandas as pd
import numpy as np
import time
import pickle
import sys
import os
from datetime import timedelta
from array import array

def init():
    global fldr
    try:
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.tsa.arima_model import ARIMA
    except ImportError:
        import pip
        package_name='statsmodels'
        pip.main(['install', package_name])
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.tsa.arima_model import ARIMA
    # serialize the model on disk in the special 'outputs' folder
    print ("Read the model from model.pkl in directory ", fldr)
    
    fl = open(fldr+"model.pkl", 'rb')
    global ar_res
    ar_res = pickle.load( fl)
    fl.close()

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

def predictForecast(strt, stp):
    import json
    import numpy
    fst = PrepareFcstData(strt,stp)

    global ar_res
    fst['forecast']=ar_res.predict(start = strt-1, end= stp, dynamic= True)
    return fst

def run(inputString):
    import json
    import numpy
    try:
        input_list=json.loads(inputString)
    except ValueError:
        return 'Bad input: expecting a json encoded list of lists.'
    strt = int(input_list[0]["start"])
    stp = int(input_list[1]["stop"])
    print("start:",strt)
    print("stop:",stp)
        
    pred = predictForecast(strt,stp)
    return str(pred)

global fldr
fldr=""
if __name__ == "__main__":
    
    fldr = os.environ['AZUREML_NATIVE_SHARE_DIRECTORY'] + "outputs/"
    # predict future values
    print ('Python version: {}'.format(sys.version))
    print('Pandas version:',pd.__version__)
    print ()
    init()
    #f = run('{"input":[{"start":"127"},{"stop":"151"}]}')
    f = run('[{"start":"127"},{"stop":"151"}]')

    print("Forecast Values:")
    print(f)


