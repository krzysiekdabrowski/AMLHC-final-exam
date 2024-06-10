import pandas as pd
from sklearn.linear_model import LogisticRegression
from logic.functions import evaluate_model


def logistic_regression(X, y, X_train, X_test, y_train, y_test):
    # create and train the logistic regression model
    log_reg = LogisticRegression(max_iter=10000)
    log_reg.fit(X_train, y_train)

    # predict the labels for the test set
    y_pred = log_reg.predict(X_test)

    # model evaluation
    results = evaluate_model(y_test, y_pred)

    # most predictive feature
    feature_names = X.columns.tolist()
    coefficients_df = pd.DataFrame({'Coefficient': log_reg.coef_[0], 'Feature': feature_names})
    sorted_coefficients = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)

    return results, sorted_coefficients

