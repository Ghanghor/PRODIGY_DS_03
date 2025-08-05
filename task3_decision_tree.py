import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

# --- Main Script ---
try:
    # 1. LOAD THE UPLOADED DATASET
    # This now points to your specific file.
    # It assumes the file is in the same folder as the script.
    file_name = 'bank-additional.csv'
    # This dataset also uses semicolons (;) as separators
    df = pd.read_csv(file_name, sep=';')
    
    print(f"--- Dataset '{file_name}' loaded successfully ---")
    print("First 5 rows of the dataset:")
    print(df.head())

    # 2. DATA PREPROCESSING
    print("\n--- Preprocessing Data ---")
    # The target variable 'y' is 'yes' or 'no'
    df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)

    # Convert all categorical columns to numerical format
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Data converted to numerical format.")

    # 3. DEFINE FEATURES (X) AND TARGET (y)
    X = df_processed.drop('y', axis=1)
    y = df_processed['y']

    # 4. SPLIT DATA INTO TRAINING AND TESTING SETS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # 5. BUILD AND TRAIN THE DECISION TREE CLASSIFIER
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    print("\n--- Decision Tree model trained successfully ---")

    # 6. EVALUATE THE MODEL
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix plot saved as 'confusion_matrix.png'")

    # 7. VISUALIZE THE DECISION TREE
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, 
                   feature_names=X.columns, 
                   class_names=['No', 'Yes'], 
                   filled=True, 
                   rounded=True,
                   max_depth=3,
                   fontsize=10)
    plt.title("Decision Tree Structure (Top 3 Levels)")
    plt.savefig('decision_tree.png')
    print("\nDecision tree visualization saved as 'decision_tree.png'")

except FileNotFoundError:
    print(f"Error: '{file_name}' not found.")
    print("Please make sure your data file and this Python script are in the exact same folder.")
except Exception as e:
    print(f"An error occurred: {e}")