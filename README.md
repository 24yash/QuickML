# QuickML: Automated Machine Learning Platform

QuickML is a Flask web application designed to automate machine learning workflows using scikit-learn. It provides a user-friendly interface that allows users to preprocess data, select models, and train them without needing to write any code.

## Features

- **Data Preprocessing**: QuickML allows users to upload their datasets and apply various preprocessing steps such as handling null values, scaling numerical values, and encoding categorical values.

- **Model Selection**: Users can choose from a variety of machine learning models including Linear Regression, Random Forest, K-Nearest Neighbors, and Logistic Regression.

- **Model Training**: Once a model is selected, QuickML trains the model using the preprocessed data.

- **Automated Predictions**: After training, the model can be used to make predictions on a user-uploaded test dataset. The predictions are then available for download.

## Getting Started

To get started with QuickML, clone the repository and install the required dependencies.

```bash
git clone https://github.com/yourusername/quickml.git
cd quickml
pip install -r requirements.txt
```

Then, run the following command to start the Flask application.

```bash
python app.py 
```

Open your web browser and navigate to `http://localhost:5000` to start using QuickML.

## Usage

1. **Upload your dataset**: On the home page, upload your training dataset and optionally a test dataset. Specify the target column in your dataset.

2. **Preprocess your data**: Choose how you want to handle null values, scale numerical values, and encode categorical values.

3. **Select and train a model**: Choose a machine learning model and train it on your preprocessed data.

4. **Make predictions**: If you uploaded a test dataset, the trained model will make predictions on this data. You can download the predictions as a CSV file.

## Tech Stack

- **Flask**: A lightweight WSGI web application framework for Python.
- **scikit-learn**: A machine learning library for Python.
- **Pandas**: A data analysis and manipulation library for Python.
- **NumPy**: A library for adding support for arrays and matrices, along with a large collection of high-level mathematical functions.
