dvc get https://github.com/iterative/dataset-registry tutorials/versioning/data.zip

Expand-Archive -Path data.zip
rm -Force data.zip

REM Linux & MacOS
REM unzip -q data.zip
REM rm -f data.zip
