import pickle
import pandas as pd
import numpy as np

# Importing machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def TrainModel(clean_data_dir,model_dir):
    print("Training Model")
    # Load the previously saved data and labels
    data_labels = pickle.load(open(clean_data_dir, 'rb'))

    # Initialize variables to keep track of rows and columns processed
    row_, col_ = 0, 0
    data_list = []

    # Convert the loaded data into a structured format
    for row in range(len(data_labels['data'])):
        temp = []  # Temporary list to hold a single row of data

        # Copy each element of the row into the temporary list
        for col in range(len(data_labels['data'][row])):
            temp.append(data_labels['data'][row][col])
            row_ = row
            col_ = col

        # Add the processed row to the main data list
        if len(temp) == 84:  # Check if the row length matches the expected size
            print((data_labels['labels'][row]))
        data_list.append(temp)

    # Print the last processed row and column index for verification
    print("Training data dimension")
    print(f"last row index: {row_}")
    print(f"last col index: {col_}")

    # Convert the processed data and labels into NumPy arrays
    data = np.asarray(data_list)
    labels = np.asarray(data_labels['labels'])

    # Create a DataFrame from the data for easier manipulation
    data_df = pd.DataFrame(data)
    labels_df = pd.DataFrame(labels)

    # Append labels to the data DataFrame
    data_df['Labels'] = labels

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

    # Initialize the label encoder
    label_encoder = LabelEncoder()
    # Encode the string labels to numerical labels
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Initialize, train and make prediction the RandomForestClassifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_predicted_RFC = model.predict(X_test)

    # Initialize, train and make prediction the XGBoost model
    XGBoost_model = XGBClassifier()
    XGBoost_model.fit(X_train, y_train_encoded)
    y_predicted_XGBoost = XGBoost_model.predict(X_test)

    # Initialize, train and make prediction the SVM model
    SVM_model = SVC()
    SVM_model.fit(X_train, y_train)
    y_predicted_svm = SVM_model.predict(X_test)

    # Calculate and print the accuracy of the model
    acc_score_RFC = accuracy_score(y_predicted_RFC, y_test)
    acc_score_XGB = accuracy_score(y_predicted_XGBoost, y_test_encoded)
    acc_score_SVM = accuracy_score(y_predicted_svm, y_test)
    print("Accuracy of Random Forest classifier:", acc_score_RFC * 100)
    print("Accuracy of XGB classifier:", acc_score_XGB * 100)
    print("Accuracy of SVM classifier:", acc_score_SVM * 100)

    # Serialize and save the trained model
    with open(model_dir, "wb") as file:
        pickle.dump({"model": model}, file)
