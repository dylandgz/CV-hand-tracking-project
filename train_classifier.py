import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

data_labels = pickle.load(open('./data.pickle', 'rb'))

row_, col_ = 0, 0
data_list = []
# print(data_list)
for row in range(len(data_labels['data'])):
    temp = []
    # print(data_labels['data'][row])
    for col in range(len(data_labels['data'][row])):
        temp.append(data_labels['data'][row][col])
        row_ = row
        col_ = col
    # print(len(temp))
    if len(temp)==84:
        print((data_labels['labels'][row]))
    data_list.append(temp)

print("row " + str(row_))
print("col " + str(col_))


data = np.asarray(data_list)
labels = np.asarray(data_labels['labels'])

# print(labels)

data_df = pd.DataFrame(data)

labels_df = pd.DataFrame(labels)

data_df['Labels'] = labels

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)

model = RandomForestClassifier()

model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print(y_predicted)
print(y_test)

acc_score = accuracy_score(y_predicted, y_test)
print("Accuracy of image classifier:", acc_score * 100)



file = open("model.pickle", "wb")
pickle.dump({"model": model}, file)
file.close()