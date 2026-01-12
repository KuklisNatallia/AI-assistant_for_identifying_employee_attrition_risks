from .shap import shap_analyz
from .hr_recommend import RecommendSystem
from .simpleLLM import SimpleOllamaEngine
from .judge import SimpleJudge

# Анализ сотрудников прогнозируемых к увольнению
def analyze_at_risk(model, X_test, threshold=0.7):
    probabilities = model.predict_proba(X_test)[:, 1]
    at_risk_mask = probabilities >= threshold
    X_test_risk = X_test[at_risk_mask]
    probabilities_risk = probabilities[at_risk_mask]

    if len(X_test_risk) == 0:
        return []

    print(f"Сотрудники с риском увольнения (порог: {threshold:.0%}): {len(X_test_risk)}")

    results = []
    messages_list = []

    for i in range(len(X_test_risk)):
        idx = X_test_risk.index[i]
        prob = probabilities_risk[i]

        #print(f"Сотрудник {i + 1} (ID: {idx})")
        #print(f"Вероятность увольнения: {probabilities_risk[i]:.1%}")

        # SHAP анализ
        risk_factors = shap_analyz(model, X_test_risk, i)
        factor_names = [f[0] for f in risk_factors]

        #print(f"Факторы риска: {factor_names}")

        # Рекомендации
        rag_system = RecommendSystem()
        rag_recs = rag_system.get_recommendations(factor_names)

        llm_engine = SimpleOllamaEngine()
        personalized_recs = llm_engine.generate_recommendations(
            factor_names,
            rag_recs,
            probabilities_risk[i]
        )

        # print("Рекомендации:")
        # for rec in personalized_recs:
        #    print(f"{rec}")

        # Оценка
        judge = SimpleJudge()
        evaluation = judge.evaluate_recommendations(
            personalized_recs,
            factor_names,
            probabilities_risk[i]
        )

        # Собираем сообщения для этого сотрудника
        employee_messages = []

        attempt = 1
        while evaluation['score'] < 6 and attempt <= 3:
            message=f"  Попытка {attempt}: оценка {evaluation['score']}/10 слишком низкая, перегенерируем..."

            print(f"  {message}")
            employee_messages.append(message)

            # Генерируем новые рекомендации с улучшенным промптом
            # improved_prompt = self._create_improved_prompt(factor_names, probabilities_risk[i], attempt)
            personalized_recs = llm_engine.generate_recommendations(
                factor_names,
                rag_recs,
                probabilities_risk[i]
                # ,improved_prompt
            )

            # Переоцениваем
            evaluation = judge.evaluate_recommendations(
                personalized_recs,
                factor_names,
                probabilities_risk[i]
            )
            attempt += 1

        final_score_message = f"Оценка: {evaluation['score']}/10"
        print(final_score_message)
        employee_messages.append(final_score_message)

        if evaluation['score'] >= 6:
            print("Рекомендации:")
            for rec in personalized_recs:
                print(f"{rec}")

        else:
            warning_message = "Внимание: рекомендации получили низкую оценку и требуют ручной доработки HR-специалистом."
            print(warning_message)

            for rec in personalized_recs:
                print(f"{rec}")

        result = {
            'employee_id': str(idx),
            'probability': float(probabilities_risk[i]),  # Числовое значение вероятности
            'probability_percent': f"{probabilities_risk[i]*100:.1f}%",  # Процент для отображения
            'risk_factors': factor_names[:3] if factor_names else [],  # Топ-3 факторов
            'recommendations': personalized_recs[:3] if personalized_recs else [],  # Топ-3 рекомендаций
            'evaluation_score': float(evaluation['score']),  # Числовая оценка
            'requires_hr_review': evaluation['score'] < 6,  # Флаг необходимости доработки
            'messages': employee_messages  # Сообщения для отладки
        }

        results.append(result)

    return results
