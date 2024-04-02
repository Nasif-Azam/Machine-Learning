# Step 1: Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load and preprocess the dataset
# Load your dataset here, replace 'X' with your feature matrix and 'y' with your target variable.
# Make sure 'X' contains only numeric features and 'y' contains the corresponding labels.
# Example: X, y = load_your_dataset()

# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# Display the first few rows of the dataset
print(df.head())

# Data Preprocessing for the Titanic dataset
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode()[0])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Extract the Features (X) and Target Variable (y)
# Extract feature matrix 'X' and target variable 'y'
X = df.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'Embarked'])
y = df['Survived']
df = df.select_dtypes(include='number')
# Display the processed data
print("Processed Feature Matrix (X):")
print(X.head())
print("\nTarget Variable (y):")
print(y.head())

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Preprocess the data and create the ANN model
# Feature scaling (optional but recommended for ANN models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the ANN model
ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)

# Step 5: Train the ANN model using the training data
ann_model.fit(X_train_scaled, y_train)

# Step 6: Evaluate the model's performance using the testing data
# Make predictions on the test data.
y_pred = ann_model.predict(X_test_scaled)

# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Display the classification report and confusion matrix.
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the learning curves
training_sizes = np.linspace(0.1, 1.0, 10)
train_acc = []
test_acc = []

for size in training_sizes:
    # Calculate the new training size for each iteration
    new_size = int(size * X_train_scaled.shape[0])

    # Subset the training data
    X_train_subset = X_train_scaled[:new_size]
    y_train_subset = y_train[:new_size]

    # Create a new ANN model for each iteration
    model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', random_state=42)
    model.fit(X_train_subset, y_train_subset)

    # Evaluate the model on training and testing data
    y_train_pred = model.predict(X_train_subset)
    y_test_pred = model.predict(X_test_scaled)

    train_acc.append(accuracy_score(y_train_subset, y_train_pred))
    test_acc.append(accuracy_score(y_test, y_test_pred))

# Plot the learning curves
plt.figure(figsize=(8, 6))
plt.plot(training_sizes * 100, train_acc, label='Training Accuracy', marker='o')
plt.plot(training_sizes * 100, test_acc, label='Testing Accuracy', marker='o')
plt.xlabel("Training Size (%)")
plt.ylabel("Accuracy")
plt.title("Learning Curves")
plt.legend()
plt.grid()
plt.show()