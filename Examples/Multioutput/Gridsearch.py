import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold


def gridsearch(X, y, model, grid,
               scoring_functions=None,
               pipeline=None,
               random_state=None,
               n_cv_general=10,
               n_cv_intrain=10,
               title='none'):

    cv_results_test = np.zeros((n_cv_general, 1))
    cv_results_generalization = np.zeros((n_cv_general, 1))
    best_index_time = np.zeros((n_cv_general, 2))
    bestparams = []
    cv_results = []
    pred = np.zeros_like(y)
    rmse = np.zeros((n_cv_general, y.shape[1]))

    kfold_gen = KFold(n_splits=n_cv_general,
                      random_state=random_state,
                      shuffle=True)

    for cv_i, (train_index, test_index) in enumerate(kfold_gen.split(X, y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        kfold = KFold(n_splits=n_cv_intrain,
                      random_state=random_state,
                      shuffle=True)

        estimator = model if pipeline is None else Pipeline(
            [pipeline, ('reg', model)])

        grid_search = GridSearchCV(estimator, grid,
                                   cv=kfold,
                                   scoring=scoring_functions,
                                   refit=True,
                                   return_train_score=False)

        grid_search.fit(x_train, y_train)

        pred[test_index] = grid_search.predict(x_test)

        rmse[cv_i, ] = np.sqrt(np.average(
            (y_test - pred[test_index])**2, axis=0))

        bestparams.append(grid_search.best_params_)

        grid_search.cv_results_[
            'final_test_error'] = grid_search.score(x_test, y_test)

        cv_results.append(grid_search.cv_results_)

        cv_results_test[cv_i, 0] = grid_search.cv_results_[
            'mean_test_score'][grid_search.best_index_]
        cv_results_generalization[cv_i, 0] = grid_search.cv_results_[
            'final_test_error']
        best_index_time[cv_i, 0] = grid_search.cv_results_[
            'mean_fit_time'][grid_search.best_index_]
        best_index_time[cv_i, 1] = grid_search.cv_results_[
            'mean_score_time'][grid_search.best_index_]

    results = {}
    results['Metric'] = [
        'Score' if scoring_functions is None else scoring_functions]
    results['Mean_test_score'] = np.mean(
        cv_results_test, axis=0)
    results['Std_test_score'] = np.std(
        cv_results_test, axis=0)
    results['Mean_generalization_score'] = np.mean(
        cv_results_generalization, axis=0)
    results['Std_generalization_score'] = np.std(
        cv_results_generalization, axis=0)
    results['mean_fit_time'] = np.mean(
        best_index_time[:, 0], axis=0)
    results['std_fit_time'] = np.std(
        best_index_time[:, 0], axis=0)
    results['mean_score_time'] = np.mean(
        best_index_time[:, 1], axis=0)
    results['std_scoret_time'] = np.std(
        best_index_time[:, 1], axis=0)

    pd.DataFrame(results).to_csv(
        title + '_Summary.csv')
    pd.DataFrame(cv_results).to_csv(
        title + '_CV_results.csv')
    pd.DataFrame(bestparams).to_csv(
        title + '_Best_Parameters.csv')
    pd.DataFrame(best_index_time, columns=["Fit_time", "Score_time"]).to_csv(
        title + 'Best_Index_time.csv')

    np.savetxt(title + 'predicted_values.csv', pred, delimiter=',')
    rm = {}
    rm['mean '] = np.mean(rmse, axis=0)
    rm['std '] = np.std(rmse, axis=0)

    pd.DataFrame(rm, index=[list('target-' + str(i)
                 for i in range(y.shape[1]))]).to_csv(title + 'RMSE_' + 'score.csv')
