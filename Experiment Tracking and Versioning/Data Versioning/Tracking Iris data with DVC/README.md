# Data Versioning with DVC 


## Project Setup
First, initialize the project. I use `--subdir` since my project is inside a parent repository.
```
dvc init --subdir
```

If the project is not in a repository:
```
git init                                # initialize Git epository
dvc init                                # initialize dvc
```


## Versioning ✍️

<br>
<h3 style="color:teal; font-weight:bold">First Experiment</h3>
<br>

**1. Get the data**

1. Download the first version (v1) of the data: [data/iris/v1](https://github.com/BecayeSoft/Machine-Learning/tree/main/Experiment%20Tracking%20and%20Versioning/data/iris/v1).
2. Store it a in folder called `data` at the root of your project.

It should look like this:
```
Tracking Iris data with DVC/
    data/
        iris.data
```

**2. Capture the state of the data**
```
dvc add data
```

**3. Train the model and capture its state**
The script `train.py` trains a Random Forest model then saves it along with some metrics.
```
python train.py
dvc add model.joblib
```

**4. Commit the current state to Git**
```
git add data.dvc model.joblib.dvc metrics.csv .gitignore
git commit -m "First model, trained with 150 rows from iris data"
git tag -a "v1.0" -m "model v1.0, 150 rows"
```


<br>
<h3 style="color:teal; font-weight:bold">Second experiment 2</h3>
<br>

**1. Get new version of the data**

Download the version 2 of the data: [data/iris/v2](https://medium.com/r/?url=https%3A%2F%2Fgithub.com%2FBecayeSoft%2FMachine-Learning%2Fblob%2Fmain%2FExperiment%2520Tracking%2520and%2520Versioning%2Fdata%2Firis%2Fv2%2Firis.data).

**2. Capture the new state of the data**
```
dvc add data
```

**3. Retrain the model and capture its new state**
```
python train.py
dvc add model.h5
```

**4. Commit the state**
```
git add data.dvc model.h5.dvc metrics.csv
git commit -m "Second model, trained with 2000 images"
git tag -a "v2.0" -m "model v2.0, 2000 images"
```

That's it! Now, we have tracked the second version of the data.


<br>
<h3 style="color:teal; font-weight:bold">Switching between workspace versions</h3>
<br>

**Checkout to a whole workspace**
```
git checkout v1.0
dvc checkout
```

`git checkout` will switch to the v1.0 of the workspace. So the .dvc files will also be retrograded.
Then  `dvc checkout` will capture the current state of the workspace to make sure it is synced with Git.

**Checkout to a specific data version**
Maybe you just need a particular version of the data:
```
git checkout v1.0 data.dvc
dvc checkout data.dvc
```

**Going back to the current version**
```
git checkout master
dvc checkout
dvc remove model.h5.dvc
```


## Automating tracking ⚙️

`dvc` add makes sense when we want to track files that come from source projects such as the files in our previousdata folder. But there are files that are the result of running some code. In our case model.joblib and metrics.csv .
A better way to capture dynamically generated files is to use `dvc stage add`.

**Create a training pipeline**
```
dvc stage add -n train 
    -d train.py -d data 
    -o model.joblib 
    -M metrics.csv 
    python train.py
```

This command creates a pipeline called `train`.
* -n: name of the pipeline
* -d: the dependencies. Here it's the script `train.py` and the data folder `data`.
* -o: the outputs. The files generated after the training.
* -M: the file to log the performance metrics.
* `python train.py`: the command to run. 

To log the performance metrics of the model in `metrics.csv`, just write them in it.
```
with open("metrics.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["accuracy", "loss"])
    writer.writerow([accuracy, loss])
```

**Run the training pipeline**
```
dvc repro
```

`dvc repro` runs the pipeline defined in the project. It will only execute the components of the pipeline that have changed.


## References
> https://dvc.org/doc/use-cases/versioning-data-and-models/tutorial