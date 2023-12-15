import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Importing machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to convert classification report to DataFrame
def report_to_df(report):
    # Split the report into lines and then into elements
    lines = report.split('\n')
    report_data = []
    for line in lines[2:-4]:  # Skip the header and the last few lines
        row_data = line.split()
        report_data.append(row_data)

    # Convert to DataFrame
    column_headers = ['class', 'precision', 'recall', 'f1-score', 'support']
    df = pd.DataFrame.from_records(report_data, columns=column_headers, exclude=['support'])
    df = df.set_index('class')
    df = df.apply(pd.to_numeric, errors='coerce')
    return df
def TrainModel(feature_clean_data_dir, model_dir):
    print("Training Model")

    # Load the saved extracted features and labels
    data_labels = pickle.load(open(feature_clean_data_dir, 'rb'))

    # Initialize variable to keep track of 2d data
    data_list = []

    # Convert the loaded data into a 2D List for use
    for row in range(len(data_labels['data'])):
        temp = []  # Temporary list to hold a single row of data

        # Copy each element of the row into the temporary list
        for col in range(len(data_labels['data'][row])):
            temp.append(data_labels['data'][row][col])

        data_list.append(temp)

    # Convert the processed data and labels into NumPy arrays
    data = np.asarray(data_list)
    labels = np.asarray(data_labels['labels'])

    # -------------------------------------
    # implement k fold stratified cross validation for even class distribution
    # Number of folds
    n_splits = 5

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits)

    # Lists to store results
    acc_scores_RFC = []
    acc_scores_XGB = []
    acc_scores_SVM = []
    conf_matrices_RFC = []
    conf_matrices_XGB = []
    conf_matrices_SVM = []
    class_reports_RFC = []
    class_reports_XGB = []
    class_reports_SVM = []

    for train_index, test_index in skf.split(data, labels):
        # Splitting data into training and testing sets for each fold
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Encode labels for XGBoost
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # RandomForestClassifier
        model_RFC = RandomForestClassifier()
        model_RFC.fit(X_train_scaled, y_train)
        y_predicted_RFC = model_RFC.predict(X_test_scaled)
        acc_scores_RFC.append(accuracy_score(y_test, y_predicted_RFC))
        conf_matrices_RFC.append(confusion_matrix(y_test, y_predicted_RFC))
        class_reports_RFC.append(classification_report(y_test, y_predicted_RFC))

        # XGBClassifier
        model_XGB = XGBClassifier()
        model_XGB.fit(X_train_scaled, y_train_encoded)
        y_predicted_XGB = model_XGB.predict(X_test_scaled)
        acc_scores_XGB.append(accuracy_score(y_test_encoded, y_predicted_XGB))
        conf_matrices_XGB.append(confusion_matrix(y_test_encoded, y_predicted_XGB))
        class_reports_XGB.append(classification_report(y_test_encoded, y_predicted_XGB))

        # SVC
        model_SVM = SVC()
        model_SVM.fit(X_train_scaled, y_train)
        y_predicted_svm = model_SVM.predict(X_test_scaled)
        acc_scores_SVM.append(accuracy_score(y_test, y_predicted_svm))
        conf_matrices_SVM.append(confusion_matrix(y_test, y_predicted_svm))
        class_reports_SVM.append(classification_report(y_test, y_predicted_svm))

    # Aggregate and print results
    print("Average Accuracy of Random Forest Classifier:", np.mean(acc_scores_RFC) * 100)
    print("Average Accuracy of XGB Classifier:", np.mean(acc_scores_XGB) * 100)
    print("Average Accuracy of SVM Classifier:", np.mean(acc_scores_SVM) * 100)

    # Aggregate confusion matrices
    agg_conf_matrix_RFC = np.array(conf_matrices_RFC).sum(axis=0)
    agg_conf_matrix_XGB = sum(conf_matrices_XGB)
    agg_conf_matrix_SVM = sum(conf_matrices_SVM)

    # Print matrices
    print("Aggregated Confusion Matrix for Random Forest Classifier: ", agg_conf_matrix_RFC)
    print("Aggregated Confusion Matrix for XGB Classifier: ", agg_conf_matrix_XGB)
    print("Aggregated Confusion Matrix for SVM Classifier: ", agg_conf_matrix_SVM)

    # Define class labels
    class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                    'U', 'V', 'W', 'X', 'Y']

    # Prepare to siplay image
    fig, ax = plt.subplots(figsize=(20, 20))

    # Display the image
    cax = ax.matshow(agg_conf_matrix_RFC, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Aggregated Confusion Matrix for Random Forest Classifier')
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels)
    plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels)

    # Annotate each cell with the numeric value using white or black depending on the background
    for i in range(agg_conf_matrix_RFC.shape[0]):
        for j in range(agg_conf_matrix_RFC.shape[1]):
            ax.text(j, i, format(agg_conf_matrix_RFC[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if agg_conf_matrix_RFC[i, j] > agg_conf_matrix_RFC.max() / 2 else "black")

    # Show the plot
    plt.show()

    # Process and aggregate classification reports for the Forest Classifier
    df_list_RFC = []
    for report in class_reports_RFC:
        df_report = report_to_df(report)
        df_list_RFC.append(df_report)

    # Concatenate all DataFrames in the list to create a single DataFrame
    df_reports_RFC = pd.concat(df_list_RFC)

    # Compute the average of the classification reports
    avg_report_RFC = df_reports_RFC.groupby(level=0).mean()

    # XGBoost Classifier
    df_list_XGB = []
    for report in class_reports_XGB:
        df_report = report_to_df(report)
        df_list_XGB.append(df_report)

    df_reports_XGB = pd.concat(df_list_XGB)
    avg_report_XGB = df_reports_XGB.groupby(level=0).mean()

    # Support Vector Machine Classifier
    df_list_SVM = []
    for report in class_reports_SVM:
        df_report = report_to_df(report)
        df_list_SVM.append(df_report)

    df_reports_SVM = pd.concat(df_list_SVM)
    avg_report_SVM = df_reports_SVM.groupby(level=0).mean()

    print("Average Classification Report for Random Forest Classifier: ", avg_report_RFC)
    print("Average Classification Report for XGB Classifier: ", avg_report_XGB)
    print("Average Classification Report for SVM Classifier: ", avg_report_SVM)

    # Save the trained model and scaler
    with open(model_dir, "wb") as file:
        pickle.dump({"model": model_RFC, "scaler": scaler}, file)
