import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score


def select_features_lasso(X, y, alpha=1.0, random_state=42):
    """
    Отбор признаков с помощью Lasso L1 регуляризации.
    Признаки с нулевыми весами исключаются.
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_array = X
    
    lasso = Lasso(alpha=alpha, random_state=random_state, max_iter=10000)
    lasso.fit(X_array, y)
    
    weights = dict(zip(feature_names, lasso.coef_))
    selected_features = [name for name, weight in weights.items() if abs(weight) > 1e-6]
    
    return {
        'selected_features': selected_features,
        'weights': weights,
        'n_selected': len(selected_features),
        'n_total': len(feature_names)
    }


def get_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """
    Вычисление важности признаков через permutation importance.
    DataFrame : Таблица с важностью признаков
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    importance_df.index = importance_df.index + 1
    
    return importance_df


def get_shap_values(model, X, n_samples=100, random_state=42):
    """
    Вычисление SHAP значений для интерпретации модели.
    Используется KernelExplainer для универсальности.
    DataFrame : Таблица со средними SHAP значениями по признакам
    """
    import shap
    
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_array = X.values
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X_array = X
    
    # Background data для KernelExplainer
    if X_array.shape[0] > n_samples:
        np.random.seed(random_state)
        indices = np.random.choice(X_array.shape[0], n_samples, replace=False)
        background_data = X_array[indices]
    else:
        background_data = X_array
    
    # Используем KernelExplainer (работает с любой моделью)
    explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(X_array[:100])
    
    # Средние абсолютные SHAP значения
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': mean_shap
    })
    
    shap_df = shap_df.sort_values('shap_value', ascending=False)
    shap_df = shap_df.reset_index(drop=True)
    shap_df.index = shap_df.index + 1
    
    return shap_df


def compare_selection_methods(X, y, model=None, alpha=1.0, n_repeats=10, random_state=42):
    """
    Сравнение методов отбора признаков: Lasso, Permutation, SHAP.
    """
    from sklearn.linear_model import Ridge
    
    # Lasso
    lasso_result = select_features_lasso(X, y, alpha=alpha, random_state=random_state)
    lasso_df = pd.DataFrame([
        {'feature': name, 'weight': weight}
        for name, weight in lasso_result['weights'].items()
    ])
    lasso_df = lasso_df.sort_values('weight', key=abs, ascending=False)
    lasso_df = lasso_df.reset_index(drop=True)
    lasso_df.index = lasso_df.index + 1
    
    # Permutation Importance
    if model is None:
        model = Ridge(alpha=1.0, random_state=random_state)
        model.fit(X, y)
    
    perm_df = get_permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
    
    # SHAP
    shap_df = get_shap_values(model, X, random_state=random_state)
    
    return {
        'lasso': lasso_df,
        'permutation': perm_df,
        'shap': shap_df
    }