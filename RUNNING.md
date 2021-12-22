Please navigate to the frontend folder, make sure to use npm to install all the dependencies mentioned in the "package.json" file, and enter command "npm run dev", a browser window should pop up shortly afterwards.

We have also stored our final DecisionTreeClassifier model inside of "machinlearning" folder with name: "OverUnderSamplePipelineModel", feel free to use .load() to function to access it and test it out.

All the intermediate results are stored in MongoDB cluster, please contact us for the link if you are interested in the content inside of the cluster.

To test some of the intermediate results, please navigate to the python scripts in separate folders and file names partitioned by functionality and feature and run "python filename.py". Please note some files need to read the original "**US_Accidents_Dec20_updated.csv**" file in order to perform some cleaning based on it and this should produce "**Accident_No_NA.csv**" which is needed by many visualization and transformation scripts. "**US_Accidents_Dec20_updated.csv**" is ignored in gitignore due to its big size.Therefore, please download that file from Kaggle and place it under the root folder, and run "**clean_na.py**" before other python script.
