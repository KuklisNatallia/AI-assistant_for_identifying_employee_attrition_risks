from .simpleLLM import SimpleOllamaEngine
import requests

class SimpleJudge:
    def __init__(self):
        self.ollama_engine = SimpleOllamaEngine()

    def evaluate_recommendations(self, recommendations, risk_factors, probability):
        # Оцениваем качество рекомендаций

        if not self.ollama_engine.is_available():
            print('LLM для оценки недоступен')
            return self._get_fallback_evaluation(recommendations)

        prompt = f'''
        Оцени эти рекомендации по удержанию сотрудника (вероятность увольнения: {probability:.1%}):

        Рекомендации:
        {chr(10).join([f'- {rec}' for rec in recommendations])}

        Факторы риска: {', '.join(risk_factors)}

        Оцени по шкале от 1 до 10 и дай краткий отзыв.
        Формат: Оценка: X/10. Отзыв: [текст]
        '''

        try:
            payload = {
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(self.ollama_engine.api_url, json=payload, timeout=30)
            result = response.json()

            return self._parse_evaluation(result['response'])

        except Exception as e:
            print(f'Ошибка при оценке рекомендаций: {e}')
            return self._get_fallback_evaluation(recommendations)

    def _parse_evaluation(self, response_text):
        # Парсим оценку от Judge
        lines = response_text.split('\n')
        evaluation = {
            'score': 5.0,
            'feedback': 'Рекомендации достаточные'
        }

        for line in lines:
            if 'Оценка:' in line:
                try:
                    # Ищем число в строке типа "Оценка: 8/10"
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        evaluation['score'] = int(numbers[0])
                except:
                    pass
            elif 'Отзыв:' in line:
                evaluation['feedback'] = line.split('Отзыв:')[-1].strip()

        return evaluation

    def _get_fallback_evaluation(self, recommendations):
        # Резервная оценка
        return {
            'score': 5.5,
            'feedback': 'Рекомендации соответствуют запросу'
        }