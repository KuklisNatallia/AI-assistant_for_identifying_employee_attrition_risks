class RecommendSystem:
    def __init__(self):
        self.knowledge_base = self._create_knowledge_base()

    def _create_knowledge_base(self):
        # Создаем базу знаний
        return {
            'OverTime': [
                'Предложить гибкий график работы',
                'Снизить количество сверхурочных часов',
                'Внедрить компенсацию за переработки'
            ],
            'MonthlyIncome': [
                'Рассмотреть повышение зарплаты',
                'Предложить бонус за результаты',
                'Пересмотреть систему компенсаций'
            ],
            'YearsSinceLastPromotion': [
                'Обсудить карьерные перспективы',
                'Создать план профессионального развития',
                'Предложить участие в новых проектах'
            ],
            'JobSatisfaction': [
                'Провести встречу для обратной связи',
                'Улучшить условия работы',
                'Предложить менторскую поддержку'
            ],
            'WorkLifeBalance': [
                'Внедрить возможность удаленной работы',
                'Предложить дополнительные дни отпуска',
                'Организовать программы оздоровления'
            ]
        }

    def get_recommendations(self, risk_factors):
        # Получаем рекомендации на основе факторов риска
        recommendations = []

        for factor in risk_factors[:2]:  # Берем до 2 факторов
            if factor in self.knowledge_base:
                recommendations.extend(self.knowledge_base[factor][:2])  # По 2 рекомендации на фактор

        # Убираем дубликаты
        return list(set(recommendations))[:3]  # 3 рекомендации