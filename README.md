# Sentiment Analysis Project

This is a sentiment analysis project built using Python and machine learning techniques. The model is trained to classify movie reviews as either positive or negative. It uses a dataset of movie reviews and applies Natural Language Processing (NLP) techniques for preprocessing and feature extraction, then trains a machine learning model to predict sentiment.

## Project Description

In this project, I implemented a sentiment analysis system that categorizes movie reviews into two sentiment classes: **positive** and **negative**. The system preprocesses the reviews using the `nltk` library, applies TF-IDF vectorization to convert text data into numerical features, and trains a machine learning model using the processed data.

The project includes the following steps:
- Loading and preprocessing text data
- Vectorizing text using `TfidfVectorizer`
- Splitting the data into training and test sets
- Training a model
- Evaluating model performance

## Dataset

The dataset used for this project comes from the [Stanford Sentiment Treebank](https://ai.stanford.edu/~amaas/data/sentiment/), which contains movie reviews categorized as positive and negative. The dataset is split into two directories: `pos` for positive reviews and `neg` for negative reviews.

- Dataset source: [Stanford Sentiment Treebank](https://ai.stanford.edu/~amaas/data/sentiment/)
- Training data path: `data/aclImdb/train`

## Requirements

To run this project, you'll need to install the following Python libraries:
- `sklearn`
- `nltk`
- `pandas`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
