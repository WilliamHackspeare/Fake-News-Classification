# Fake News Classification using Machine Learning

## Table of Contents
- [Introduction](#introduction)
- [Technologies](#technologies)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)

## Introduction
This project aims to build a machine learning model to accurately classify news articles as either 'FAKE' or 'REAL'. The project employs text cleaning, feature engineering, and Random Forest Classifier for achieving high classification accuracy. Dataset taken from https://www.kaggle.com/datasets/rchitic17/real-or-fake

## Technologies
- Python 3.7
- Pandas
- Scikit-learn
- NLTK

## Features
- Text Cleaning: Removal of special characters, stopwords, and lemmatization.
- Text Vectorization: Using Term Frequency-Inverse Document Frequency (TF-IDF).
- Classification: Utilizes Random Forest Classifier for news classification.
- Evaluation: Metrics such as accuracy, precision, recall, and F1-score are used for model evaluation.

## Installation
1. Clone the repository to your local machine.
   ```
   git clone https://github.com/WilliamHackspeare/Fake-News-Classification.git
   ```
2. Navigate to the project directory.
   ```
   cd Fake-News-Classification
   ```
3. Install the required packages.
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Python script to train the model and evaluate its performance.
   ```
   python Fake_News_Classification.py
   ```

## Results
- Model Accuracy: 91.2%
- Precision, Recall, and F1-score: Above 90% for both 'FAKE' and 'REAL' classes.
