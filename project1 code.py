import pandas as pd

# Load the dataset (replace 'dataset.csv' with your actual dataset file)
data = pd.read_csv(r'C:\Users\vishn\OneDrive\Desktop\intrn\microfinance_data.csv')

# Explore the dataset
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Get information about the dataset (data types, missing values, etc.)
print(data.describe())  # Statistical summary of the numerical features
# Drop irrelevant columns
data.drop(columns=['msisdn', 'pcircle', 'pdate'], inplace=True)

# Handle missing values if any
data.dropna(inplace=True)

# Encode categorical variables if any
# If 'label' is categorical, encode it using label encoding or one-hot encoding

# Feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['label']))
data_scaled = pd.DataFrame(scaled_features, columns=data.columns[:-1])  # Exclude 'label'

# Split the data into features and target variable
X = data_scaled  # Features
y = data['label']  # Target variable
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)  # You can tune the hyperparameters further
model.fit(X_train, y_train)
# Evaluate the model on the test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Log Loss:", logloss)
