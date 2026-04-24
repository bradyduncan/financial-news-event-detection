from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    # Run without xgboost for runtime reasons
    HAS_XGBOOST = False


def run_grid_search(model, param_grid, X_train, y_train, seed):
    # Shared CV helper
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=cv,
        n_jobs=-1,
        scoring="f1_macro",
        verbose=0,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_logistic_regression(X_train, y_train, seed):
    # Train baseline LR with grid search
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        random_state=seed,
    )
    grid = {"C": [0.1, 1.0, 10.0]}
    return run_grid_search(model, grid, X_train, y_train, seed)


def train_linear_svm(X_train, y_train, seed):
    # Linear SVM with grid search
    model = LinearSVC(random_state=seed)
    grid = {"C": [0.1, 1.0, 10.0]}
    return run_grid_search(model, grid, X_train, y_train, seed)


def train_xgboost(X_train, y_train, seed):
    # Return None when xgboost is unavailable
    if not HAS_XGBOOST:
        return None, None, None
    num_classes = len(set(y_train))
    # XGBoost with grid search 
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=seed,
    )
    grid = {
        "n_estimators": [100, 300],
        "max_depth": [3, 6],
        "learning_rate": [0.05, 0.1],
    }
    return run_grid_search(model, grid, X_train, y_train, seed)
