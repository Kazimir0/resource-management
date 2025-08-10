import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

# Select features
features = [
    'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism',
    'Handcap', 'SMS_received', 'DaysBetween'
]
X = data[features]
y = data['No-show']

# Stratified train-test split to keep class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: scaler + RandomForest (with balanced class weights)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Hyperparameter grid for tuning
param_grid = {
    'clf__n_estimators': [100, 200],  # poți adăuga 300 dacă vrei
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}

# GridSearchCV cu scoring f1_macro (bine pentru clase dezechilibrate)
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

# Antrenează modelul cu tunarea hiperparametrilor
grid_search.fit(X_train, y_train)

# Cel mai bun model după tuning
best_model = grid_search.best_estimator_

# Predicții pe test
y_pred = best_model.predict(X_test)

# Rapoarte evaluare
print("Best parameters found:", grid_search.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Salvează modelul antrenat
joblib.dump(best_model, '../no_show_model.pkl')
print("\nModel salvat în ../no_show_model.pkl")
