## Train model


The statsmodels package makes training ARIMA based forecasting model really simple with few lines of code.

Run etstrain.py in a local Docker container.
```
$ az ml experiment submit -t docker etstrain.py
```

Download model from the outputs folder by clicking **outputs/model.pkl** 


## Score model


Run score.py in a local docker container
```
$ az ml experiment submit -t docker  score.py
```

## Deploy model

Now you have scoring file, model, and all dependencies to deploy model into production as web service. We will use DSVM and ssh client Mobaxterm to deploy model into DSVM. Use [this](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro) link to provision [DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro). Use [MobaXterm](http://mobaxterm.mobatek.net/download.html) from [here](http://mobaxterm.mobatek.net/download.html) as ssh client.

Copy scoring.py and model.pkl files to the DSVM using [WinSCP](https://winscp.net/eng/download.php) or other ssh ftp client. 

Now execute the following commands on DSVM to setup the local environment for operationalization
```
$ az ml env setup # this will setup local environment (not K8 cluster)
$ sudo /opt/microsoft/azureml/initial_setup.sh #add your user name to docker in root mode
```
Once environment is primed using above commands, we can start publishing as web service using the following commands
```	
$ az ml env local # target the local environment
$ az ml service create realtime -f score.py -m model.pkl  -n arimaforecast -r scikit-py -l
```
## Consuming model using Web Service

Once the web service is published, user can consume this web service using the following command
```
$ az ml service run realtime -n arimaforecast -d '{"input":[{"start":"127"},{"stop":"151"}]}'
```
