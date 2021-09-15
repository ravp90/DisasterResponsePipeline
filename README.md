# Disaster Response Pipeline

## Introduction 
This repository contains a deployable web app for a disaster response pipeline. The purpose of this app is to allow a user to enter the text from a social media post and the output from the app is a classification result which classifies if the post is indicating a disaster has occurred. 
Social media is so widely used that a disaster can often be reported first on social media before it is reported in more traditional media or before emergency services are contacted. This type of app could modified to ingest high volumes of social media post to find locations of disasters as early as possible, with the aim to respond to these disasters as soon as possible. 

## Project Contents
The project structure is as follows:
* app folder
	* templates folder
		* go.html
		* master.html
	* run.py script (deploy web app)
* data folder
	* DisasterResponse.db database
	* disaster_categories.csv data file
	* disaster_messages.csv data file
	* process_data.py script (ETL)
* models
	* classifier.pkl model
	* train_classifier.py script (ML pipeline)
* README.md - this file
* requirements.txt 

## How to deploy the app
The app is ready for deployment already, the repository contains a populated database and pickled model that can be run immediately in a web app. 

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        - `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        - `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    - `python run.py`

3. Go to http://0.0.0.0:3001/

### Using the app
The main page for the app looks as follows:
![Landing Page](screenshots/LandingPage.PNG?raw=True)

Enter a message into the message box:
![Enter a message](screenshots/EnterAMessage.PNG?raw=true)

Then click the Classify Message. The classes will then appear in the list highlighted in green:
![Classification](screenshots/Classification.PNG?raw=true)

## Authors
Ravi Parekh, MEng PhD. 

## License
This project is created using some content (data and code) provided by Udacity under the Data Scientist Nanodegree. A lot of the code is customised by the author of this repository. The author declares that the customised code is free to use and waives all copyright or related rights to this work. 

## Acknowledgements
* [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)
* [Appen](https://appen.com/)
