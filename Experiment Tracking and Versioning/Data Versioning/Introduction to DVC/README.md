# Data Versioning with DVC 

Datasets and ML model getting started
[versioning tutorial](https://dvc.org/doc/tutorials/versioning).

## How DVC works

**dvc add command**
```
dvc add data
```

Tells Git to ignore the directory `data/` and puts it into the cache (while keeping a file link to it in the workspace, so you can continue working the same way as before).

### How DVC track changes
DVC does not directly commit files to Git, it store them in cache (in dvc/cache), add them to .gitignore.

**Then, how are the files tracked?**

DVC creates .dvc files (e.g. data.dvc) which contain a reference to the cached file.
This is because Git won't accept large files such as big datasets.

<br>
Time to practice!

<br>

## Project Setup
```
git clone https://github.com/iterative/example-versioning.git
cd example-versioning

python3 -m venv .env
source .env/bin/activate                # Windows: .\.env\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Versioning ✍️

<br>
<h3 style="color:teal; font-weight:bold">Version 1</h3>
<br>

**1. Get the data**
```
dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip

# Linux & MacOS
unzip -q data.zip
rm -f data.zip

# Windows
Expand-Archive -Path data.zip
rm -Force data.zip
```

**2. Capture the state of the data**
See *How DVC track changes* above.
```
dvc add data
```

**3. Train the model and capture its state**
```
python train.py
dvc add model.h5
```

**4. Commit the current state**
```
git add data.dvc model.h5.dvc metrics.csv .gitignore
git commit -m "First model, trained with 1000 images"
git tag -a "v1.0" -m "model v1.0, 1000 images"
```


<br>
<h3 style="color:teal; font-weight:bold">Version 2</h3>
<br>

**1. Get new version of the data**
```
dvc get https://github.com/iterative/dataset-registry/tutorials/versioning/new-labels.zip
dvc get https://github.com/iterative/dataset-registry/tutorials/versioning

# Linux & MacOS
unzip -q new-labels.zip
rm -f new-labels.zip

# Windows
Expand-Archive -Path new-labels.zip
Expand-Archive -Path "new-labels.zip" -DestinationPath ".\data"
rm -Force new-labels.zip
```

**2. Capture the new state of the data**
```
dvc add data
```

**3. Retrain the model capture its new state**
```
python train.py
dvc add model.h5
```
**4. Commit the current state**
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
Then  `dvc checkout` will capture the current state of the workspace to make sure it is sync with Git.

**Checkout to a specific data version**
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

**Create a training pipeline**
```
dvc stage add -n train -d train.py -d data \
        -o model.h5 -o bottleneck_features_train.npy \
        -o bottleneck_features_validation.npy \
        -M metrics.csv
        python train.py
```

This command creates a pipeline called `train`.
* -n: name of the pipeline
* -d: the dependencies. Here it's the script `train.py` and the data folder `data`.
* -o: the outputs. The files generated after the training.
* -M: the file to log the performance metrics.
* `python train.py`: the command to run. 


**Run the training pipeline**
```
dvc repro
```

`dvc repro` runs the pipeline defined in the project. It will only execute the components of the pipeline that have changed.

## References
> https://dvc.org/doc/use-cases/versioning-data-and-models/tutorial