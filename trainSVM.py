import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


dataFrame = pd.read_csv('Features/train/fea_all.csv')
X = dataFrame.iloc[:, :-1].values 
y_train = dataFrame.iloc[:, -1].values  
X_train = dataFrame.drop(['Image_name', 'Type_x'], axis=1)


# dataFrame1 = pd.read_csv('Features/validate/validate_features1.csv')
dataFrame1 = pd.read_csv('Features/validate/validate_features1.csv')
X_test = dataFrame1.iloc[:, :-1].values  
y_test = dataFrame1.iloc[:, -1].values   
X_test = dataFrame1.drop(['Image_name', 'Type_x'], axis=1)

# Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the SVM model
svm_model = SVC(kernel='poly', C=1.5, gamma='scale')  

# Training
svm_model.fit(X_train, y_train)

# Predict
y_pred = svm_model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()

print(f"True Positives: {TP}")
print(f"False Positives: {FP}")
print(f"True Negatives: {TN}")
print(f"False Negatives: {FN}")

print(classification_report(y_test, y_pred))
