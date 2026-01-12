import requests
import json


class SimpleOllamaEngine:
    def __init__(self):
        self.api_url = 'http://localhost:11434/api/generate'
        self.model = 'qwen2.5:1.5b'

    def is_available(self):
        # Проверяем доступность Ollama
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            return response.status_code == 200
        except:
            return False

    def generate_recommendations(self, risk_factors, rag_recommendations, probability):
        # Генерируем рекомендации через Ollama

        if not self.is_available():
            print('Ollama недоступен. Используем базовые рекомендации.')
            return self._get_fallback_recommendations(risk_factors, rag_recommendations)

        prompt = f'''
        Ты опытный HR-специалист с 10+ лет опыта. Сотрудник имеет вероятность увольнения {probability:.1%}.
        Факторы риска: {', '.join(risk_factors)}

        ЗАДАЧА: Предложи 3 КОНКРЕТНЫХ, ПРАКТИЧЕСКИ РЕАЛИЗУЕМЫХ рекомендации для удержания.

        ОГРАНИЧЕНИЯ:
        1. НЕ предлагать снижать зарплату, ухудшать условия или понижать в должности.
        2. Рекомендации должны быть СОГЛАСОВАНЫ с факторами риска.
        3. Каждая рекомендация должна быть ИЗМЕРИМА и КОНКРЕТНА.
        4. Максимальная длина каждой рекомендации - 15 слов.

        Ответ в формате:
        1. Рекомендация 1
        2. Рекомендация 2  
        3. Рекомендация 3

        ПРИМЕР ХОРОШИХ РЕКОМЕНДАЦИЙ:
        1. Внедрить гибкий график работы с 2 днями удаленки в неделю
        2. Предложить курс повышения квалификации за счет компании
        3. Назначить наставника для карьерного роста в течение месяца
        '''

        try:
            payload = {
                'model': self.model,
                'prompt': prompt,
                'stream': False
            }

            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()  # Проверяем HTTP ошибки
            result = response.json()

            # return self._parse_response(result['response'])
            if 'response' in result:
                parsed_recs = self._parse_response(result['response'])
                if parsed_recs:  # Если удалось распарсить
                    return parsed_recs

        except Exception as e:
            print(f'Ошибка при обращении к Ollama: {e}')
            return self._get_fallback_recommendations(risk_factors, rag_recommendations)

    def _parse_response(self, response_text):
        # Парсим ответ от LLM
        lines = response_text.strip().split('\n')
        recommendations = []

        for line in lines:
            line = line.strip()
            if line and (
                    line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('-')):
                # Убираем нумерацию и маркеры
                clean_line = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                # Фильтруем слишком короткие строки
                if clean_line and len(clean_line) > 10:
                    recommendations.append(clean_line)

        return recommendations[:3] if recommendations else []

    def _get_fallback_recommendations(self, risk_factors, rag_recommendations):
        # Резервные рекомендации
        return [
            'Провести индивидуальную встречу для обсуждения карьерных перспектив',
            'Пересмотреть текущую нагрузку и рабочие задачи',
            'Предложить участие в программе профессионального развития']