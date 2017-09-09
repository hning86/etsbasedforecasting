
## Pre-requisites
Install stats package. Open command line from Azure Machine Learning Workbench and execute the following statement

```
$ conda install statsmodels
```

## Train model


The statsmodels package makes training ARIMA based forecasting model really simple with few lines of code.

Run etstrain.py in a local Docker container.
```
$ az ml experiment submit   -c local etstrain.py
```

Download model from the outputs folder by clicking **outputs/model.pkl** 


## Score model


Run score.py in a local docker container
```
$ az ml experiment submit   -c local score.py
```

## Deploy model

Now you have scoring file, model, and all dependencies to deploy model into production as web service. Open command line interface through File Menu of Azure Machine Learning Workbench App. 

Now execute the following commands on CLI to setup the local environment for operationalization assuming you already have Azure subscription and Azure Machine Learning model management account
```
$ az account set -s <Subscription Name e.g. mysubscription>
$ az ml account modelmanagement set -n <acct name e.g. neerajteam2hosting> -g <rsrc grp e.g. amlgrp2>
```

If you have already Azure Machine Learning model manamgement environment, then just set the environment as shown below

```
$ az ml  env set -n <env name e.g. amlcluster> -g <rsrc grp e.g. amlgrp2>
```

If you don't have existing Azure Machine Learning model management environment, then you can create a new environment using the following command. The command below is shown without **-c option** to setup a local environment for ease of testing. If you need to setup a ACS Kubernetes cluster, then also add **-c option** to the env setup command below. 

```
$ az ml env setup -n <acct name e.g. neerajteam2hosting> -g <rsrc grp e.g. amlgrp2>
```
Once environment is primed using above commands, we can start publishing as web service using the following commands
```	
$ az ml env local # target the local environment
$ az ml service create realtime -f score.py -m model.pkl  -n arimaforecast -r python -c .\aml_config\conda_dependencies.yml -l true
```
## Consuming model using Web Service

Once the web service is published, user can consume this web service using the following command
```
# for local
az ml service run realtime -i <webservice name e.g. arimaforecast> -d "[{\"start\":\"127\"},{\"stop\":\"151\"}]"
# for cluster 
$ az ml service run realtime -i <webservice FQDN e.g. arimaforecast.amlcluster-41011325.eastus2> -d "[{\"start\":\"127\"},{\"stop\":\"151\"}]"
```
