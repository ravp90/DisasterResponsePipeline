# Disaster Response Pipeline

## Introduction 
This repository contains a deployable web app for a disaster response pipeline. The purpose of this app is to allow a user to enter the text from a social media post and the output from the app is a classification result which classifies if the post is indicating a disaster has occurred. 
Social media is so widely used that a disaster can often be reported first on social media before it is reported in more traditional media or before emergency services are contacted. This type of app could modified to ingest high volumes of social media post to find locations of disasters as early as possible, with the aim to respond to these disasters as soon as possible. 

## Project Contents
In the root of the repository, there are three folders and 2 files, one of the files is this README file. There is a requirements.txt file which includes all the python modules and versions that were used to build this response pipeline. There is a data folder, a models folder and an app folder. 

The data folder contains two CSV files for disaster messages and disaster categories, a labelled dataset of messages for training and validating the model. The process_data.py file is the file used to ingest the data, clean it and save it into the DisasterResponse.db database, ready for modelling. 

The models folder contains a train_classifier.py file which loads the data saved in the database, pre-processes the data into usable inputs and outputs for classification models in sci-kit learn and then splits the data into a train and test set. The file also contains a pipeline for processing the text data using a tokenizer passed into CountVectorizer, a TF-IDF transformation and finally a multi-class classifier which uses Adaboost. The pipeline also has a gridsearch algorithm applied to it with different learning rates and numbers of estimators in the ensemble. 
The model identified from the gridsearch is used to evaluate the test data and output a classification report for each class, and the model is saved as a pickled file in this same folder, named classifier.pkl. 

Finally the app folder contains a run.py script which has some plotly charts and a flask applet which is deployed using the model and the training data. The web app has an entry field where the user can enter a social media post in plain-text and the output shows a list of classes, where if the post is classified into one or more classes the app highlights those classes in green. 

## How to deploy the app
The app is ready for deployment already, the repository already contains a populated database and pickled model that can be run immediately in a web app. 
The user must first install the modules from the requirements.txt file, then run the run.py file in the app folder. 

To run the full pipeline from ETL to ML and then deploy the web app, the user must again install the modules from the requirements.txt file, then from the data folder, run the process_data.py script. Subsequently, run the train_classifier.py script from the models folder, and finally run the run.py script in the app folder. 

## License
This project is created using some content (data and code) provided by Udacity under the Data Scientist Nanodegree. A lot of the code is customised by the author of this repository. The author declares that the customised code is free to use and waives all copyright or related rights to this work. 