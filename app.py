from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'abcd'

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        target = request.form['target']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        file_path = file.filename

        test_file = request.files['test_file']
        test_file_path = None
        if test_file:
            test_file_path = os.path.join(app.config['UPLOAD_FOLDER'], test_file.filename)
            test_file.save(test_file_path)
            test_file_path = test_file.filename
        
        data = pd.read_csv('uploads/' + file_path)

        if target not in data.columns:
            flash(f'Target column "{target}" does not exist in the uploaded file.', 'error')
            return redirect(url_for('index')) 

        # Identify target variable type
        target_type = 'categorical' if data[target].dtype == 'object' else 'numerical'
        print(target_type)
        # Separate columns and calculate null and unique values
        num_cols = data.select_dtypes(include='number').columns.tolist()
        cat_cols = data.select_dtypes(include='object').columns.tolist()
        
        null_values = data.isnull().sum().to_dict()

        unique_values = data.nunique().to_dict()

        return render_template('columns.html', num_cols=num_cols, cat_cols=cat_cols, target=target, target_type=target_type, null_values=null_values, unique_values=unique_values, filename=file_path, test_file_path=test_file_path)

def apply_preprocessing(filename, options, target, test_file_path):
    data = pd.read_csv('uploads/' + filename)
    test = pd.read_csv('uploads/' + test_file_path)  
    target_column = target 
    exclude_columns = ['id', 'ID', 'Id']


    # Ensure the target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Handle null values for numerical columns
    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col not in exclude_columns]  # Exclude specified columns
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)  # Exclude the target column from preprocessing
    
    if options['null_handling'] == 'drop':
        data.dropna(subset=numerical_columns, inplace=True)
    elif options['null_handling'] == 'mean':
        for col in numerical_columns:
            data[col].fillna(data[col].mean(), inplace=True)
    elif options['null_handling'] == 'median':
        for col in numerical_columns:
            data[col].fillna(data[col].median(), inplace=True)
    elif options['null_handling'] == 'constant':
        data[numerical_columns].fillna(options['null_constant'], inplace=True)
    
    # Handle null values for categorical columns
    categorical_columns = data.select_dtypes(include='object').columns.tolist()
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]  # Exclude specified columns
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    if options['null_handling_categorical'] == 'drop':
        data.dropna(subset=categorical_columns, inplace=True)
    elif options['null_handling_categorical'] == 'mode':
        for col in categorical_columns:
            data[col].fillna(data[col].mode()[0], inplace=True)
    elif options['null_handling_categorical'] == 'constant':
        data[categorical_columns].fillna(options['null_categorical_constant'], inplace=True)
    
    # Scaling
    if options['scaling'] == 'standard':
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        test[numerical_columns] = scaler.fit_transform(test[numerical_columns])
    elif options['scaling'] == 'minmax':
        scaler = MinMaxScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        test[numerical_columns] = scaler.fit_transform(test[numerical_columns])

    if options['categorical_handling'] == 'onehot':
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
        test = pd.get_dummies(test, columns=categorical_columns, drop_first=True)
        # Align the test dataset columns with the training dataset
        # Drop the target column from data.columns
        data_columns_without_target = data.columns.drop(target_column)
        # Reindex test with the modified columns
        test = test.reindex(columns=data_columns_without_target, fill_value=0)

    elif options['categorical_handling'] == 'label':
        print('label encoding')
        label_encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            test[column] = le.transform(test[column])
            label_encoders[column] = le  # Store label encoder for each column if needed later

    if data[target_column].dtype == 'object':
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        joblib.dump(le, f'{target_column}_label_encoder.pkl')
        print(f"Encoded target column '{target_column}' with LabelEncoder.")
    
    data.to_csv('uploads/' + filename, index=False)
    test.to_csv('uploads/' + test_file_path, index=False)
    print("Preprocessing complete")


@app.route('/process_data/<filename>/<target>/<test_file_path>', methods=['GET', 'POST'])
def process_data(filename, target, test_file_path):
    null_handling = request.form['null_handling']
    null_constant = request.form.get('null_constant', None)  # Optional, for 'constant' option
    null_handling_categorical = request.form['null_handling_categorical']
    null_categorical_constant = request.form.get('null_categorical_constant', 'Unknown')  # Default to 'Unknown'
    scaling = request.form['scaling']
    categorical_handling = request.form['categorical_handling']
    
    # Assuming you save these options to apply later
    options = {
        "null_handling": null_handling,
        "null_constant": null_constant,
        "null_handling_categorical": null_handling_categorical,
        "null_categorical_constant": null_categorical_constant,
        "scaling": scaling,
        "categorical_handling": categorical_handling,
    }
    
    print(filename)
    # Placeholder for where you might save these options, e.g., to a session or database
    print(options)
    apply_preprocessing(filename, options, target, test_file_path)
    
    return redirect(url_for('model_selection', filename=filename, target=target, test_file_path=test_file_path))

def train_model(model_type, filename, target):
    data = pd.read_csv('uploads/' + filename)
    X = data.drop(target, axis=1)
    y = data[target]
    
    print('training model')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model based on the selected type
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'logistic':
        model = LogisticRegression()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    
    # Save the trained model to the local directory
    model_filename = f"{model_type}_model.pkl"
    joblib.dump(model, 'uploads/predictions'+model_filename)
    return model_filename

@app.route('/model_selection/<filename>/<target>/<test_file_path>', methods=['GET', 'POST'])
def model_selection(filename, target, test_file_path):
    if request.method == 'POST':
        model_type = request.form['model']
        print(model_type)
        model_file = train_model(model_type, filename, target)
        return redirect(url_for('model_trained', model_file=model_file, filename=filename, target=target, test_file_path=test_file_path))
    return render_template('model_selection.html', filename=filename, target=target, test_file_path=test_file_path)

@app.route('/model_trained/<model_file>/<filename>/<target>/<test_file_path>', methods=['GET', 'POST'])
def model_trained(model_file, filename, target, test_file_path):
    model = joblib.load(model_file)
    
    # Load the test set
    test_filename = test_file_path
    test_data = pd.read_csv('uploads/' + test_filename)
    
    predictions = model.predict(test_data)
    
    # Check if 'Id', 'ID', or 'Id' column exists and keep it along with predictions
    id_column = None
    for possible_id_column in ['Id', 'ID', 'id']:
        if possible_id_column in test_data.columns:
            id_column = possible_id_column
            break
    
    if id_column:
        final_data = test_data[[id_column]].copy()
    else:
        final_data = pd.DataFrame(index=test_data.index)
    
    final_data['predictions'] = predictions
    
    # Save the modified test set with only Id and predictions
    final_data.to_csv('uploads/predictions' + test_file_path, index=False)
    
    return render_template('download_test_set.html', model_file=model_file, filename=filename, target=target, test_filename=test_filename)

from flask import send_from_directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route('/download/<path:filename>')
def download_file(filename):
    # Ensure the path is safe and does not navigate outside directory
    if '..' in filename or filename.startswith('/'):
        return "Invalid path", 400
    return send_from_directory(BASE_DIR, 'uploads/predictions' + filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)