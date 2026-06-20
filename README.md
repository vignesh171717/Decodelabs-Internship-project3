# AI Recommendation System Using TF-IDF

## Overview

This project is a simple AI-powered Recommendation System developed as part of the DecodeLabs Artificial Intelligence Internship (Project 3).

The system recommends relevant courses based on user interests using Natural Language Processing (NLP) techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) and Cosine Similarity.

## Features

- User-friendly command-line interface
- Displays available recommendation topics
- Converts text to lowercase for better matching
- Uses TF-IDF for feature extraction
- Uses Cosine Similarity for recommendation logic
- Recommends multiple related courses
- Handles invalid inputs gracefully

## Technologies Used

- Python
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity

## How It Works

1. The user enters an interest or topic.
2. The system preprocesses the text by converting it to lowercase.
3. TF-IDF converts the text into numerical vectors.
4. Cosine Similarity compares the user's interest with available course descriptions.
5. Relevant courses are recommended to the user.
