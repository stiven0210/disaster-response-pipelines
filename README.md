# Disaster-Response-Pipelines

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl models/vocabulary_stats.pkl models/category_stats.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Required libraries
- nltk 3.3.0
- numpy 1.15.2
- pandas 0.23.4
- scikit-learn 0.20.0
- sqlalchemy 1.2.12

## Motivation

In this project, It will provide disaster responses to analyze data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

Below are a few screenshots of the web app.

<img src="assets/disaster-response-project1.png" width="80%" alt="disaster response project web app">
<img src="assets/disaster-response-project2.png" width="80%" alt="disaster response project web app">

## Files

- ETL Pipeline Preparation.ipynb: Description for workspace/data/process_data.py
- ML Pipeline Preparation.ipynb: Description for workspace/model/train_classifier.py
- workspace/data/process_data.py: A data cleaning pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database
- workspace/model/train_classifier.py: A machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file

## Acknowledgements

I wish to thank [Figure Eight](https://www.figure-eight.com/) for dataset, and thank [Udacity](https://www.udacity.com/) for advice and review.



