# NLP Text Classifier

A simple sentiment analysis project using the IMDB movie review dataset.
The model classifies text as Positive or Negative using TF-IDF and Logistic Regression.

## How it works

1. download_data.py  
   - Downloads the IMDB dataset  
   - Saves it to data/data.csv

2. train.py  
   - Trains the TF-IDF vectorizer and Logistic Regression model  
   - Saves model.pkl and vectorizer.pkl into models/  
   - Generates confusion_matrix.png and roc_curve.png

3. predict.py  
   - Lets you type text and returns the predicted sentiment + confidence

## How to run

# Download dataset
python download_data.py

# Train the model
python train.py

# Make predictions
python predict.py

## Folder structure

data/       # dataset  
models/     # saved models + plots  
utils/      # helper code  

## License
MIT License
