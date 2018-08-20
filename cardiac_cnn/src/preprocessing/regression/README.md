This preprocessing code is heavily based on the solution by Julian de Wit. For documenation about the approach look [here](http://juliandewit.github.io/kaggle-ndsb/)

#### Dependencies & data
The dicom data needs to be downloaded from [Kaggle](https://www.kaggle.com/c/second-annual-data-science-bowl/data) and must be extracted in the data/train /validate and /test folders.

#### Run the solution 
1. *python step1_preprocess.py*<br> This will process the dicom files in the train and validate folder and output the train_enriched.csv which will be used by Step 3. 
2. *python step2_preprocess.py*<br>This will process the dicom files in the test folder and output the test_enriched.csv which will be used by Step 3.
3. Open step3.Rmd in RStudio to run the relevant data. This will later be converted to an executable R Script.  



