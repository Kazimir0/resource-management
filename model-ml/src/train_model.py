import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load data
data = pd.read_csv('../data/KaggleV2-May-2016.csv')

# Convert target column to numeric
data['No-show'] = data['No-show'].map({'No': 0, 'Yes': 1})

# Feature engineering: calculate days between scheduled and appointment
data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
data['DaysBetween'] = (data['AppointmentDay'] - data['ScheduledDay']).dt.days.clip(lower=0)

# Additional temporal features
data['AppointmentDayOfWeek'] = data['AppointmentDay'].dt.dayofweek  # 0=Mon
data['ScheduledHour'] = data['ScheduledDay'].dt.hour
data['IsWeekend'] = data['AppointmentDay'].dt.dayofweek.isin([5, 6]).astype(int)

# Select features (numeric + categorical)
numeric_features = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism',
    'Handcap', 'SMS_received', 'DaysBetween',
    'AppointmentDayOfWeek', 'ScheduledHour', 'IsWeekend'
]
categorical_features = ['Gender', 'Neighbourhood']

X = data[numeric_features + categorical_features]
y = data['No-show']

# Stratified train-test split to keep class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Persist the held-out test split (full original columns) for external evaluation
test_indices = X_test.index
data.loc[test_indices].to_csv('../data/test_split.csv', index=False)
print("Saved held-out test split to ../data/test_split.csv")

# Preprocessor: passthrough numeric, one-hot encode categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Pipeline with placeholder classifier; GridSearch will swap it
pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Hyperparameter grids: LogisticRegression and RandomForest
param_grid = [
    {
        'clf': [LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')],
        'clf__C': [0.1, 1.0, 3.0]
    },
    {
        'clf': [RandomForestClassifier(class_weight='balanced', random_state=42)],
        'clf__n_estimators': [200, 400],
        'clf__max_depth': [10, 20, None],
        'clf__min_samples_split': [2, 5],
        'clf__min_samples_leaf': [1, 2]
    }
]

# GridSearchCV with f1_macro scoring (good for imbalanced classes)
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

# Train the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Best model after tuning
best_model = grid_search.best_estimator_

# Predictions on test set
y_pred = best_model.predict(X_test)

# Evaluation reports
print("Best parameters found:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
joblib.dump(best_model, '../no_show_model.pkl')
print("\nModel saved to ../no_show_model.pkl")
