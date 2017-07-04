This project demonstrates on how to use Vienna for TimeSeries forecasting using Python library statsmodels for timeseries forecasting

##Train model
The statsmodels package makes training ARIMA based forecasting model really simple with few lines of code.

The data for this example was in the following format

    time, N1725
	1	, 3740
	2	, 2880
	...

As seen in the above example, time is not specified as datetime. Hence, we need to use python libraries to convert this to a specific timestamp.

The **PrepareTrainData** function in the training file converts each period into a hypothetical timestamp for forecasting to work. The code for prepareTrainData is as follows:

    def PrepareTrainData(dataset):
	    dataset2 = dataset[['time']] # create new DF with time period only
	    dataset3 = dataset2.astype(np.float64) #convert time period to float for deltatime function
	    
		#create new df that will save the final timestamp
		d2 = pd.DataFrame(np.zeros(shape=(len(dataset3), 1)))
	    
		#use timedelta to create a new timestamp using time period from our dataset as hour
	    for idx in range(len(dataset3)):
	    	d = timedelta(hours=dataset3.iloc[idx, 0])
	    	d2.iloc[idx, 0] = pd.datetime(2017, 6, 19, 0, 0, 0, 0) + d
	    
	    #get value portion of dataframe
	    d3 = dataset[['N1725']]
	    frames = [d2, d3]
		#combine both newly created timestamp dataframe and value dataframe
	    d4 = pd.concat(frames, axis=1)
	    
		#create the return dataframe with appropriate column names
	    d5 = pd.DataFrame({'N1725':d4.iloc[0:126,1].astype(np.float64)}) #d4.iloc[0:108,0:1]
	    d5.index = (d4.iloc[0:126,0])
	    d5.index.name = 'time'
	    return d5

Once data is prepared, training is just a single line of code as shown below

	def TrainTimeSeries(dataset, p, d, q, freq):
    	if freq > 0:
    	    model = ARIMA(dataset,order= (p,d,q))
    	    return model

Training uses additional parameters p, d, q which represents parameterizing Arima model as described below. For this example, we will use p=6, d=0, q=2.

- p is the auto-regressive part of the model. It allows us to incorporate the effect of past values into our model e.g. lag. 
- d is the integrated part of the model. This includes terms in the model that incorporate the amount of differencing (i.e. the number of past time points to subtract from the current value) to apply to the time series. 
- q is the moving average part of the model. This allows us to set the error of our model as a linear combination of the error values observed at previous time points in the past.

Once model has been trained, we serialize the model into a file for scoring and deployment.

	# serialize the model on disk in the special 'outputs' folder
	print ("Export the model to model.pkl")
	fl = open('./outputs/model.pkl', 'wb')
	pickle.dump(ar_res, fl)
	fl.close()

##Model Performance
We need to measure the quality of model from run history perspective to track ongoing performance. For this purpose, we are going to use metric RMSE (Root Mean Square Error) to evaluate performance of time series model. 

To evaluate RMSE, we will first find the fit (forecast values) of existing timestamps with model and compared with known values at those timestamps. Then, we will use model to forecast the future values.

The code to calculate RMSE and the code to fit existing values is shown below:

	def CalResultsArr(model):
	    results_AR = model.fit(disp=-1)
	    plt.plot(results_AR.fittedvalues, color='red')
	    return results_AR
	def rmse(predictions, targets):
    	return np.sqrt(((predictions - targets) ** 2).mean())
	
	trgt = dataset2['N1725'].as_matrix()
	ar_res = CalResultsArr(ar_mdl)
	pred = ar_res.fittedvalues.as_matrix()
	rmseval = rmse(pred,trgt)
	

The picture below shows original timeseries as blue, estimated timeseries as red, and forecasted timeseries for future timestamps as green. 

![](http://neerajkh.blob.core.windows.net/images/forecast.png)

##Score model
Once model has been trained, scoring script (aka score.py) needs to deserialize model and use the model for creating forecast (shown as green in the picture in previous section)

The scoring script needs to author 2 functions 

- init() - This function will install any package dependencies and deserialized model. For this example, it will install statsmodels as package and deserialize model using pickle.  

	    def init():
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
	    	print ("Read the model from model.pkl")
	    	fl = open('model.pkl', 'rb')
	    	global ar_res
	    	ar_res = pickle.load( fl)
	    	fl.close()
- run() - This function will be called on every prediction and will use model to create a forecast


 
		def run(inputString):
		    import json
		    import numpy
		    try:
		        input_list=json.loads(inputString)
		    except ValueError:
		        return 'Bad input: expecting a json encoded list of lists.'
		    strt = int(input_list["input"][0]["start"])
		    stp = int(input_list["input"][1]["stop"])
		    print("start:",strt)
		    print("stop:",stp)
		        
		    pred = predictForecast(strt,stp)
		    return json.dumps(str(pred))

The run function also need to prepare forecast data by converting start and stop time periods to time stamps. The run function then can use the predict function for generating the forecast. The following code provides details on how to convert time periods to time stamps and then use predictForecast to actually generate predictions

	
	def predictForecast(strt, stp):
	    import json
	    import numpy
	    fst = PrepareFcstData(strt,stp)
	
	    global ar_res
	    fst['forecast']=ar_res.predict(start = strt-1, end= stp, dynamic= True)
	    return fst 

##Deploying model into production as Web Service

Now you have scoring file, model, and all dependencies to deploy model into production as web service. We will use DSVM and ssh client Mobaxterm to deploy model into DSVM. Use [this](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro) link to provision [DSVM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro). Use [MobaXterm](http://mobaxterm.mobatek.net/download.html) from [here](http://mobaxterm.mobatek.net/download.html) as ssh client. 

Once DSVM is provisioned, download the model generated from the training script above into local directory and copy scoring.py and model.pkl files to the DSVM using [WinSCP](https://winscp.net/eng/download.php) or other ssh ftp client. Now the DSVM will have the following files in the desired project directory. 

	[matt@mattdsvm ETSBasedForecasting]$ pwd
	/home/matt/vienna/ETSBasedForecasting
	[matt@mattdsvm ETSBasedForecasting]$ ls
	model.pkl     score.py          

Now execute the following commands on DSVM to setup the local environment for operationalization

	az component update --add ml # this will add ml component for az cli
	az ml env setup # this will setup local environment (not K8 cluster)
	source ~/.amlenvrc # this will setup global variables
	sudo /opt/microsoft/azureml/initial_setup.sh #add your user name to docker in root mode

Once environment is primed using above commands, we can start publishing as web service using the following commands
	
	az ml env local # target the local environment
	az ml service create realtime -f score.py -m model.pkl  -n arimaforecast -r scikit-py -l

The last command publishes realtime scoring web service using score.py file that has init() and run() methods. It provides model file as part of -m parameter. It provide runtime as python using -r parameter. It enables logging to AppInsights using -l option. Finally, it names this web service as arimaforecast.

##Consuming model using Web Service

Once the web service is published, user can consume this web service using the following command

	az ml service run realtime -n arimaforecast -d '{"input":[{"start":"127"},{"stop":"151"}]}'

The above command is for forecasting values for the time periods 127-151. 

##Debugging model in production


###Cluster failing to provision
If the K8 cluster fails to provision with the following error, then check the subscription ownership, make sure you have appropriate permissions on the subscription, and reissue the command. 

	[matt@mattdsvm ETSBasedForecasting]$ az ml env setup --cluster
	Creating ssh key /home/matt/.ssh/acs_id_rsa
	Setting up your Azure ML environment with a storage account, App Insights account, ACR registry and ACS cluster.
	Enter environment name (1-20 characters, lowercase alphanumeric): mattcluster
	Subscription set to Channels Subscription
	Continue with this subscription (Y/n)? Y
	Creating resource group mattclusterrg
	Started App Insights Account deployment.
	Creating ACR registry and storage account: mattclusteracr and mattclusterstor (please be patient, this can take several minutes)
	Setting up Kubernetes Cluster in ACS.
	kubectl is not installed on the path. One click install? (Y/n): Y
	Downloading client to /home/matt/bin/kubectl from https://storage.googleapis.com/kubernetes-release/release/v1.7.0/bin/linux/amd64/kubectl
	Ensure /home/matt/bin/kubectl is on the path to avoid seeing this message in the future.
	creating service principal.........done
	waiting for AAD role to propagate.done
	Creating Kubernetes cluster. Please be patient--this could take up to ten minutes.
	Downloading kubeconfig file to /home/matt
	Authentication failed.

###Docker Container doesn't have model
When I first started executing the command to create and consume web service, the python sample on github was missing parameter to include model as part of the CLI command. The following consumption command was giving error:

	[matt@mattdsvm ETSBasedForecasting]$ az ml service run realtime -n arimaforecast-d '{"input":[{"start":"127"},{"stop":"151"}]}'
	[Local mode] No service named arimaforecast running locally.
	To run a remote service, switch environments using: az ml env remote
	Expected exactly one container with label amlid=arimaforecast and found 0.

Use the docker commands **docker logs sharp_stallman** or **docker logs <container id>** to see the logs.  


	[matt@mattdsvm ETSBasedForecasting]$ docker logs sharp_stallman
	2017-07-03 20:29:14,608 CRIT Supervisor running as root (no user in config file)
	2017-07-03 20:29:14,611 INFO supervisord started with pid 1
	2017-07-03 20:29:15,614 INFO spawned: 'rsyslog' with pid 9
	2017-07-03 20:29:15,615 INFO spawned: 'program_exit' with pid 10
	2017-07-03 20:29:15,616 INFO spawned: 'nginx' with pid 11
	2017-07-03 20:29:15,625 INFO spawned: 'gunicorn' with pid 12
	2017-07-03 20:29:16,669 INFO success: rsyslog entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
	2017-07-03 20:29:16,670 INFO success: program_exit entered RUNNING state, process has stayed up for > than 1 seconds (startsecs)
	Collecting statsmodels
	  Downloading statsmodels-0.8.0-cp35-cp35m-manylinux1_x86_64.whl (6.2MB)
	Requirement already satisfied (use --upgrade to upgrade): scipy in /usr/local/lib/python3.5/dist-packages (from statsmodels)
	Collecting patsy (from statsmodels)
	  Downloading patsy-0.4.1-py2.py3-none-any.whl (233kB)
	Requirement already satisfied (use --upgrade to upgrade): pandas in /usr/local/lib/python3.5/dist-packages (from statsmodels)
	Requirement already satisfied (use --upgrade to upgrade): numpy>=1.8.2 in /usr/local/lib/python3.5/dist-packages (from scipy->statsmodels)
	Requirement already satisfied (use --upgrade to upgrade): six in /usr/local/lib/python3.5/dist-packages (from patsy->statsmodels)
	Requirement already satisfied (use --upgrade to upgrade): pytz>=2011k in /usr/local/lib/python3.5/dist-packages (from pandas->statsmodels)
	Requirement already satisfied (use --upgrade to upgrade): python-dateutil>=2 in /usr/local/lib/python3.5/dist-packages (from pandas->statsmodels)
	Installing collected packages: patsy, statsmodels
	2017-07-03 20:29:21,309 INFO success: nginx entered RUNNING state, process has stayed up for > than 5 seconds (startsecs)
	Successfully installed patsy-0.4.1 statsmodels-0.8.0
	You are using pip version 8.1.1, however version 9.0.1 is available.
	You should consider upgrading via the 'pip install --upgrade pip' command.
	Read the model from model.pkl
	{"lineno": 253, "tags": [], "filename": "glogging.py", "level": "ERROR", "exc_info": "Traceback (most recent call last):\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/arbiter.py\", line 557, in spawn_worker\n    worker.init_process()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/workers/base.py\", line 126, in init_process\n    self.load_wsgi()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/workers/base.py\", line 136, in load_wsgi\n    self.wsgi = self.app.wsgi()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/app/base.py\", line 67, in wsgi\n    self.callable = self.load()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/app/wsgiapp.py\", line 65, in load\n    return self.load_wsgiapp()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/app/wsgiapp.py\", line 52, in load_wsgiapp\n    return util.import_app(self.app_uri)\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/util.py\", line 357, in import_app\n    __import__(module)\n  File \"/var/azureml-app/wsgi.py\", line 4, in <module>\n    import app\n  File \"/var/azureml-app/app.py\", line 5, in <module>\n    from main import *\n  File \"/var/azureml-app/main.py\", line 66, in <module>\n    init()\n  File \"/var/azureml-app/main.py\", **line 22, in init\n    fl = open('model.pkl', 'rb')\nFileNotFoundError**: [Errno 2] No such file or directory: 'model.pkl'\n", "logger": "gunicorn.error", "path": "/usr/local/lib/python3.5/dist-packages/gunicorn/glogging.py", "message": "Exception in worker process", "timestamp": "2017-07-03T20:29:22.496277Z", "host": "98a3f3a6e299", "stack_info": null}
	2017-07-03 20:29:22,637 INFO exited: gunicorn (exit status 3; not expected)
	
	2017-07-03 20:29:28,679 INFO gave up: gunicorn entered FATAL state, too many start retries too quickly
	2017-07-03 20:29:29,681 WARN program_exit: bad result line: 'Killing supervisor with this event: ver:3.0 server:supervisor serial:0 pool:program_exit poolserial:0 eventname:PROCESS_STATE_FATAL len:58'
	2017-07-03 20:29:29,681 WARN program_exit: has entered the UNKNOWN state and will no longer receive events, this usually indicates the process violated the eventlistener protocol
	2017-07-03 20:29:29,681 WARN received SIGQUIT indicating exit request
	2017-07-03 20:29:29,681 INFO waiting for nginx, rsyslog, program_exit to die
	2017-07-03 20:29:30,685 INFO stopped: nginx (exit status 0)
	2017-07-03 20:29:30,686 INFO stopped: program_exit (terminated by SIGTERM)
	2017-07-03 20:29:30,687 INFO stopped: rsyslog (exit status 0)

As you can see from the logs, model file is not present. So resubmit the image creation command with the model file as shown below:

	az ml service create realtime -f score.py -m model.pkl   -n arimaforecast -r scikit-py -l

Instead of 

	az ml service create realtime -f score.py -n arimaforecast -r scikit-py -l

###Cleaning docker images
Once you have published and created container image, then republishing and using the web service may continue to reuse previously published image and  not the recent image. Use the following commands to remove previous images

	[matt@mattdsvm ETSBasedForecasting]$ docker images
	REPOSITORY                                TAG                 IMAGE ID            CREATED             SIZE
	forecastenvacr.azurecr.io/arimaforecast   <none>              26da21a3ed9f        12 minutes ago      983 MB
	forecastenvacr.azurecr.io/arimaforecast   <none>              058faa76b3ad        47 minutes ago      983 MB
	forecastenvacr.azurecr.io/arimaforecast   <none>              b9e2f7f57b80        2 hours ago         983 MB

	[matt@mattdsvm ETSBasedForecasting]$ docker rmi forecastenvacr.azurecr.io/arimaforecast -f
	Untagged: forecastenvacr.azurecr.io/arimaforecast:latest
	Untagged: forecastenvacr.azurecr.io/arimaforecast@sha256:b1c1774da3e9d17a69a7327aa656b05c4b5ab8d1a90c7feb514194ebf03643a2
	Deleted: sha256:edf33bc49b38c3f7ec4cc6ff91e67536d9d45a669fee7aa5995443b0ea393e39

	[matt@mattdsvm ETSBasedForecasting]$ docker rmi 26da21a3ed9f -f
	Untagged: forecastenvacr.azurecr.io/arimaforecast@sha256:3ae8831cac5d13a3879c3e21afe4355d35e27525bcb59fdb96a01468c05fcfb0
	Deleted: sha256:26da21a3ed9f6b107261773b5887101c249c47fd76a9990b4d23bae4ae733849


###Incorrect package version used by the docker
Once all of the above issues were resolved, the service was still not callable and docker logs suggested that absence of module called **pandas.core.indexes**. Comparing the pandas versions of Vienna App and web service image, it was clear that Vienna App is using 0.20.2 version while web service is using 0.19.2   

    {"filename": "glogging.py", "path": "/usr/local/lib/python3.5/dist-packages/gunicorn/glogging.py", "message": "Exception in worker process", "lineno": 253, "exc_info": "Traceback (most recent call last):\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/arbiter.py\", line 557, in spawn_worker\nworker.init_process()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/workers/base.py\", line 126, in init_process\nself.load_wsgi()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/workers/base.py\", line 136, in load_wsgi\nself.wsgi = self.app.wsgi()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/app/base.py\", line 67, in wsgi\nself.callable = self.load()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/app/wsgiapp.py\", line 65, in load\nreturn self.load_wsgiapp()\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/app/wsgiapp.py\", line 52, in load_wsgiapp\nreturn util.import_app(self.app_uri)\n  File \"/usr/local/lib/python3.5/dist-packages/gunicorn/util.py\", line 357, in import_app\n__import__(module)\n  File \"/var/azureml-app/wsgi.py\", line 4, in <module>\nimport app\n  File \"/var/azureml-app/app.py\", line 5, in <module>\nfrom main import *\n  File \"/var/azureml-app/main.py\", line 66, in <module>\ninit()\n  File \"/var/azureml-app/main.py\", line 24, in init\nar_res = pickle.load( fl)\nImportError: No module named 'pandas.core.indexes'\n", "level": "ERROR", "stack_info": null, "logger": "gunicorn.error", "tags": [], "timestamp": "2017-07-03T20:52:02.852305Z", "host": "6234b9e70e48"}
 
To fix this issue, create a requirements.txt file under the same directory on DSVM as scoring and model file with the following contents

	[matt@mattdsvm ETSBasedForecasting]$ cat requirements.txt
	pandas==0.20.2

Once the above file has been created, republish the web service with the following parameters

    az ml service create realtime -f score.py -m model.pkl   -n arimaforecast2 -p requirements.txt -r scikit-py -l




###Docker logs for specific container
Many times, we need to just look at the logs from specific container rather than all docker logs. Below are the commands for retrieving specific docker logs. 

    
    
    [matt@mattdsvm ETSBasedForecasting]$ docker ps -a
    CONTAINER IDIMAGE  COMMAND  CREATED STATUS PORTS   NAMES
    6234b9e70e48forecastenvacr.azurecr.io/arimaforecast1   "supervisord -c /etc/"   5 minutes ago   Exited (0) 4 minutes ago   amazing_bassi
    efa523712ba6forecastenvacr.azurecr.io/arimaforecast"supervisord -c /etc/"   13 minutes ago  Exited (0) 13 minutes ago  cocky_goldberg
    ea156cb9c25c1b73c4d597a2   "supervisord -c /etc/"   16 minutes ago  Exited (0) 16 minutes ago  gloomy_borg
    a9e09e478a15edf33bc49b38   "supervisord -c /etc/"   20 minutes ago  Exited (0) 19 minutes ago  stupefied_swirles
    98a3f3a6e29926da21a3ed9f   "supervisord -c /etc/"   27 minutes ago  Exited (0) 27 minutes ago  sharp_stallman
    f6731299fcc6058faa76b3ad   "supervisord -c /etc/"   About an hour ago   Exited (0) About an hour ago   jovial_chandrasekhar
    a30f6341592ab9e2f7f57b80   "supervisord -c /etc/"   2 hours ago Exited (0) 2 hours ago modest_murdock
    [matt@mattdsvm ETSBasedForecasting]$ docker logs 6234b9e70e48
    2017-07-03 20:51:49,333 CRIT Supervisor running as root (no user in config file)
    2017-07-03 20:51:49,335 INFO supervisord started with pid 1


###Environment variables are incorrectly set
When I first started provisiong and setting up DSVM, I found that environment variables were not set correctly. If you see the error as shown below, then you are most likely having same issue.

    az ml service create realtime -f score.py  -n arimaforecast -r scikit-py
     
    Please set up your storage account for AML:
      export AML_STORAGE_ACCT_NAME=<yourstorageaccountname>
      export AML_STORAGE_ACCT_KEY=<yourstorageaccountkey>
     
     
    Please set up your ACR registry for AML:
      export AML_ACR_HOME=<youracrdomain>
      export AML_ACR_USER=<youracrusername>
      export AML_ACR_PW=<youracrpassword>
     
     
    Please set up your ACR registry for AML:
      export AML_ACR_HOME=<youracrdomain>
      export AML_ACR_USER=<youracrusername>
      export AML_ACR_PW=<youracrpassword>


Please run the following command again and verify that ~/.amlenvrc file exists.

	az ml env setup
##Using Appinsights for telemetry

AppInsights provides convenient way to debug your model and understand what is going on with the model in production. 
###Traces 
The traces table provides information about the requests observed by the web service such as client information, originating location etc. 

![](http://neerajkh.blob.core.windows.net/images/TraceCapture.PNG)


###Request/Response Logs
AppInsights also captures request/response logs for every call made to the webservice hosted in container as shown below.

 
![](http://neerajkh.blob.core.windows.net/images/RequestlogsCapture.PNG)
