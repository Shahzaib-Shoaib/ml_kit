# ML/NLP Competition Cheatsheet

# Importing essential libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the data
df = pd.read_csv('your_dataset.csv')

# Quick look at the data
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Split the data
X = df.drop('target_column', axis=1)
y = df['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a full pipeline with a random forest classifier
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

# Fit the pipeline
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Feature importance (for random forest)
feature_importance = clf.named_steps['classifier'].feature_importances_
feature_names = (numeric_features.tolist() + 
                 clf.named_steps['preprocessor']
                    .named_transformers_['cat']
                    .named_steps['onehot']
                    .get_feature_names(categorical_features).tolist())

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# For regression problems, use this instead:
# reg = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('regressor', GradientBoostingRegressor())])
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# print('MSE:', mean_squared_error(y_test, y_pred))
# print('R2 Score:', r2_score(y_test, y_pred))

# NLP specific preprocessing (if needed)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    
    return ' '.join(clean_tokens)

# Apply preprocessing to text column
df['clean_text'] = df['text_column'].apply(preprocess_text)

# Create BoW or TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(df['clean_text'])

# Combine with other features if needed
X_combined = np.hstack((X.values, X_text.toarray()))

# Train-test split and model fitting as before...

# Save the model
from joblib import dump
dump(clf, 'model.joblib')

# Load the model (in a new session)
# from joblib import load
# clf = load('model.joblib')
