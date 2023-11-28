import pickle
import pandas as pd
import numpy as np

# Importing machine learning tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
def TrainModel():
    print("Training Model")
    # Load the previously saved data and labels
    data_labels = pickle.load(open('./data.pickle', 'rb'))

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
    print("training data dimension")
    print("last row index " + str(row_))
    print("last col index " + str(col_))

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

    # Initialize and train the RandomForestClassifier model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_predicted = model.predict(X_test)

    # Calculate and print the accuracy of the model
    acc_score = accuracy_score(y_predicted, y_test)
    print("Accuracy of image classifier:", acc_score * 100)

    # Serialize and save the trained model
    with open("model.pickle", "wb") as file:
        pickle.dump({"model": model}, file)