import shap
from catboost import Pool


def shap_analyz(model, X_test, employee_index=0):
    # Создаем
    test_pool = Pool(X_test, cat_features=[col for col in X_test.columns if X_test[col].dtype == 'object'])

    # Получаем SHAP значения
    shap_values = model.get_feature_importance(type='ShapValues', data=test_pool)
    shap_values = shap_values[:, :-1]  # убираем базовое значение

    # Анализ конкретного сотрудника
    employee_shap = shap_values[employee_index]
    employee_data = X_test.iloc[employee_index]

    # Находим топ-3 фактора риска
    feature_impact = []
    for i, feature in enumerate(X_test.columns):
        impact = employee_shap[i]
        feature_impact.append((feature, impact, employee_data[feature]))

    # Сортируем по влиянию
    feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)

    # Возвращаем топ-3 фактора риска
    return feature_impact[:3]