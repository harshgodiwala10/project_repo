import pandas as pd
import joblib, optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report
# , mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning)


df = pd.read_csv('C:/Users/Harsh/OneDrive/Desktop/python practice/teleCust1000t.csv')

X = ['region','tenure','age','marital','address','income','ed','employ','retire','gender','reside']
y = ['custcat']
X = df[X]
y = df[y]
encoder = LabelEncoder()
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = encoder.fit_transform(y)
y = y.ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


global_model_results = pd.DataFrame(columns=[
    'model_name', 'algorithm', 'accuracy', 'f1_score',
    'classification_report', 'hyperparameters', 'model_file'
])
def train_random_forest():
    try:
        def objective(trial):
            # Suggest hyperparameters for RandomForest
            n_estimators = trial.suggest_int('n_estimators', 10, 100)  # Number of trees
            max_depth = trial.suggest_int('max_depth', 3, 20)  # Max depth of trees
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)  # Minimum samples required to split a node
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)  # Minimum samples required at leaf node
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])  # Feature selection for splits

            # Create and train the RandomForest model with the suggested hyperparameters
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on accuracy
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Access the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_random_forest_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"RandomForest_trial_{trial.number}",
                    "algorithm": "RandomForest",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 RandomForest Models stored successfully!")

    except Exception as e:
        print(f"Error in RandomForest training: {str(e)}")


def train_decision_tree():
    try:
        def objective(trial):
            # Suggest hyperparameters for Decision Tree
            max_depth = trial.suggest_int('max_depth', 1, 50)  # Maximum depth of the tree
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)  # Minimum samples required to split
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  # Minimum samples at a leaf node
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])  # Split quality criterion

            # Create and train the Decision Tree model with the suggested hyperparameters
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                random_state=42
            )

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the best hyperparameters and top 10 trials
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Add results to the global DataFrame
        global global_model_results  # Access the global DataFrame

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = DecisionTreeClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_decision_tree_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"DecisionTree_trial_{trial.number}",
                    "algorithm": "DecisionTree",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 Decision Tree Models stored successfully!")

    except Exception as e:
        print(f"Error in Decision Tree training: {str(e)}")


def train_knn():
    try:
        def objective(trial):
            # Suggest hyperparameters for KNN
            n_neighbors = trial.suggest_int('n_neighbors', 1, 50)  # Number of neighbors
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Weight function
            p = trial.suggest_int('p', 1, 2)  # Distance metric (1: Manhattan, 2: Euclidean)

            # Create and train the KNN model with the suggested hyperparameters
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the best hyperparameters and top 10 trials
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Access the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = KNeighborsClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_knn_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"KNN_trial_{trial.number}",
                    "algorithm": "KNN",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 KNN Models stored successfully!")

    except Exception as e:
        print(f"Error in KNN training: {str(e)}")



def train_svc():
    try:
        def objective(trial):
            # Suggest hyperparameters for SVC
            C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Regularization parameter
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])  # Kernel type
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])  # Kernel coefficient

            # Create and train the SVC model with the suggested hyperparameters
            model = SVC(C=C, kernel=kernel, gamma=gamma)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, n_jobs=-1, cv=3)  # 3-fold cross-validation
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on accuracy
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Access the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = SVC(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file
            model_filename = f"best_svc_model_trial_{trial.number}.pkl"
            # joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"SVC_trial_{trial.number}",
                    "algorithm": "SVC",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 SVC Models stored successfully!")

    except Exception as e:
        print(f"Error in SVC training: {str(e)}")



def train_logistic_regression():
    try:
        def objective(trial):
            # Suggest hyperparameters for Logistic Regression
            C = trial.suggest_loguniform('C', 1e-3, 1e3)  # Regularization strength
            solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'sag', 'saga'])  # Solver type
            max_iter = trial.suggest_int('max_iter', 100, 1000, step=100)  # Maximum iterations

            # Create and train the Logistic Regression model with suggested hyperparameters
            model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)

            # Perform cross-validation to evaluate the model
            score = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=3, n_jobs=-1)  # 3-fold CV
            return score.mean()  # Return mean cross-validation score

        # Create the Optuna study and optimize the objective function
        study = optuna.create_study(direction='maximize')  # Maximizing accuracy
        study.optimize(objective, n_trials=100)  # Perform 100 trials for hyperparameter tuning

        # Get the top 10 trials based on accuracy
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:10]

        # Update the global DataFrame
        global global_model_results

        for trial in best_trials:
            params = trial.params
            # Train the model with the best hyperparameters
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            # Evaluate the model on the test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)

            # Save the model to file (optional)
            model_filename = f"best_logistic_model_trial_{trial.number}.pkl"
            # Uncomment to save: joblib.dump(model, model_filename)

            # Append details to the global DataFrame
            global_model_results = pd.concat([
                global_model_results,
                pd.DataFrame([{
                    "model_name": f"LogisticRegression_trial_{trial.number}",
                    "algorithm": "Logistic Regression",
                    "accuracy": accuracy,
                    "f1_score": f1,
                    "classification_report": classification_rep,
                    "hyperparameters": params,
                    "model_file": model_filename
                }])
            ], ignore_index=True)

        print("Top 10 Logistic Regression Models stored successfully!")

    except Exception as e:
        print(f"Error in Logistic Regression training: {str(e)}")
# if __name__ == '__main__':

if __name__ == "__main__":
    train_knn()
    train_svc()
    train_logistic_regression()
    train_decision_tree()
    train_random_forest()
    print(global_model_results.head())
