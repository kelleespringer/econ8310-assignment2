import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score  # Import accuracy_score for evaluation

# Load training and test data
training_data_url = 'https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv'
test_data_url = 'https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv'
train_df = pd.read_csv(training_data_url)
test_df = pd.read_csv(test_data_url)

# Combine training and test data for consistent encoding
combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
# Identify non-numeric columns
non_numeric_columns = combined_df.select_dtypes(include=['object']).columns

# Convert non-numeric columns using Label Encoding
label_encoders = {}
for col in non_numeric_columns:
    label_encoders[col] = LabelEncoder()
    combined_df[col] = label_encoders[col].fit_transform(combined_df[col])

# Split the combined data back into train and test sets
train_df = combined_df.xs('train')
test_df = combined_df.xs('test')

# Features and target variable
X = train_df.drop('meal', axis=1)
y = train_df['meal']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (choose one)
#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = DecisionTreeClassifier(random_state=42)
#model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train the model
model.fit(X_train, y_train)

# Save the fitted model
modelFit = model

# Remove 'meal' column from test data
test_features = test_df.drop('meal', axis=1)

# Make predictions
pred = modelFit.predict(test_features)

# Convert predictions to binary values (1 or 0)
pred = [int(p) for p in pred]

# Evaluate the model (optional, for your own reference)
val_predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, val_predictions)
print(f'Validation Accuracy: {accuracy:.2f}')
