import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler


def calculate_metrics(y_true, y_pred):
    """
    Вычисляет все метрики качества модели.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name=None):
    """
    Оценивает модель на train и test выборках.
    """
    # Предсказания
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Метрики
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Gap (разница между train и test)
    gap = {metric: train_metrics[metric] - test_metrics[metric] 
           for metric in train_metrics.keys()}
    
    result = {
        'model_name': model_name or type(model).__name__,
        'train': train_metrics,
        'test': test_metrics,
        'gap': gap
    }
    
    return result


def create_cv_splits(n_splits=5, method='kfold', groups=None, target=None):
    """
    Создает стратегии кросс-валидации.
    """
    if method == 'kfold':
        return KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    elif method == 'stratified':
        if target is None:
            raise ValueError("target required for StratifiedKFold")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    elif method == 'group':
        if groups is None:
            raise ValueError("groups required for GroupKFold")
        return GroupKFold(n_splits=n_splits)
    
    elif method == 'time':
        return TimeSeriesSplit(n_splits=n_splits)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def cross_validate(model, X, y, n_splits=5, method='kfold', 
                   groups=None, target=None, scoring='neg_mean_absolute_error'):
    """
    Выполняет кросс-валидацию модели.
    """
    from sklearn.model_selection import cross_val_score
    
    cv = create_cv_splits(n_splits=n_splits, method=method, 
                          groups=groups, target=target)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    return {
        'mean_score': -np.mean(scores),  # Для MAE меняем знак
        'std_score': np.std(scores),
        'scores': -scores
    }


def evaluate_splits(splits, df, features, target_col, model=None):
    """
    Оценивает модель на кастомных разбиениях (как в ML3.ipynb).
    """
    if model is None:
        model = Ridge(alpha=1.0)
    
    scores = []
    
    for train_idx, test_idx in splits:
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target_col]
        X_test = df.loc[test_idx, features]
        y_test = df.loc[test_idx, target_col]
        
        # Масштабирование
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение и предсказание
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Метрика
        mae = mean_absolute_error(y_test, y_pred)
        scores.append(mae)
    
    return np.mean(scores), np.std(scores)


def compare_cv_methods(df, features, target_col, n_splits=5, groups_col=None):
    """
    Сравнивает различные стратегии CV (как в ML3.ipynb).
    """
    cv_results = {}
    
    # Custom K-Fold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    custom_kf_splits = [(list(train), list(test)) for train, test in kf.split(df)]
    cv_results['Custom K-Fold'] = evaluate_splits(custom_kf_splits, df, features, target_col)
    
    # Custom Stratified (если target категориальный)
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        custom_strat_splits = [(list(train), list(test)) for train, test in 
                               skf.split(df, df[target_col])]
        cv_results['Custom Stratified'] = evaluate_splits(custom_strat_splits, df, features, target_col)
    except:
        pass
    
    # Custom Time Series
    tscv = TimeSeriesSplit(n_splits=n_splits)
    custom_time_splits = [(list(train), list(test)) for train, test in tscv.split(df)]
    cv_results['Custom Time Series'] = evaluate_splits(custom_time_splits, df, features, target_col)
    
    # Custom Group K-Fold
    if groups_col is not None:
        gkf = GroupKFold(n_splits=n_splits)
        custom_group_splits = [(list(train), list(test)) for train, test in 
                               gkf.split(df, groups=df[groups_col])]
        cv_results['Custom Group'] = evaluate_splits(custom_group_splits, df, features, target_col)
    
    # Sklearn встроенные методы (для сравнения)
    model = Ridge(alpha=1.0)
    X = df[features].values
    y = df[target_col].values
    
    # Sklearn Stratified
    try:
        sklearn_strat = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        sklearn_strat_scores = cross_val_score(model, X, y, cv=sklearn_strat, 
                                               scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_results['Sklearn Stratified'] = (-np.mean(sklearn_strat_scores), np.std(sklearn_strat_scores))
    except:
        pass
    
    # Sklearn Group
    if groups_col is not None:
        sklearn_group = GroupKFold(n_splits=n_splits)
        sklearn_group_scores = cross_val_score(model, X, y, cv=sklearn_group,
                                               scoring='neg_mean_absolute_error', n_jobs=-1,
                                               groups=df[groups_col].values)
        cv_results['Sklearn Group'] = (-np.mean(sklearn_group_scores), np.std(sklearn_group_scores))
    
    # Создаем DataFrame
    cv_df = pd.DataFrame({
        'Method': list(cv_results.keys()),
        'MAE ($)': [v[0] for v in cv_results.values()],
        'Std ($)': [v[1] for v in cv_results.values()]
    })
    
    return cv_df