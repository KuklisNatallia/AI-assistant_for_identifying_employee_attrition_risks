from datetime import datetime
from typing import List, Dict, Optional
from sqlmodel import SQLModel, Field, Relationship
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, precision_recall_curve, auc
#import datetime as dt
from catboost import CatBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import json
import joblib
from my_first_project.appp.models.balance import BalanceService
from my_first_project.appp.models.user import User


def preprocessing():
    data = pd.read_csv('WA_Fn_UseC_HR_Employee_Attrition.csv')
    df = data.copy()
    df = df.drop(columns=['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'],
                    errors='ignore')  # Удаляем константы и неинформативные столбцы
    df['Attrition'] = df['Attrition'].astype(str).replace({'Yes': 1, 'No': 0}).astype(int)
    return df

# Обучим CatBoost
def model_catboost(save_model=False, model_path="catboost_model.joblib"):
    df = preprocessing()

    # Подготовим данные для моделирования и разделим на test и train
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)

    # Балансировка классов
    rus = RandomUnderSampler(random_state=42)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)  # или model_catboost.load_model('model.cbm')

    # Создание модели
    catboost_model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.05,
        depth=6,  # глубина, увеличили с 4
        verbose=False,  # 100,
        random_seed=42,
        # auto_class_weights='Balanced',
        l2_leaf_reg=3,  # добавили регуляризацию
        border_count=50,
        rsm=0.8,
        eval_metric='Recall',
        early_stopping_rounds=50  # Предотвращение переобучения
    )

    # Обучение модели
    begin_time = datetime.now()
    catboost_model.fit(
        X_rus, y_rus,
        eval_set=(X_test, y_test),
        cat_features=[col for col in X.columns if X[col].dtype == 'object'],
        use_best_model=True,
        verbose=50
    )
    training_time = datetime.now() - begin_time

    # Предсказания
    y_pred_test = catboost_model.predict(X_test)
    y_pred_train = catboost_model.predict(X_rus)
    y_pred_proba = catboost_model.predict_proba(X_test)[:, 1]  # вероятности для положительного класса

    # Метрики
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)

    # AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall_curve, precision_curve)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"Время обучения: {training_time}")

    # Сохранение модели если нужно
    if save_model:
        joblib.dump(catboost_model, model_path)
        print(f"Модель сохранена в {model_path}")

    # Возвращаем результаты
    catboost_results = {
        'model': catboost_model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_pr': auc_pr,
        'training_time': training_time,
        'X_test': X_test,
        'y_test': y_test
    }

    return catboost_results

# Класс для работы с уже обученной моделью
class TrainedModel:
    def __init__(self, model_path: str = "catboost_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        # Загружаем сохраненную модель
        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            print(f"Модель загружена из {self.model_path}")
        except FileNotFoundError:
            print(f"Файл модели {self.model_path} не найден")
            print("Сначала обучите модель с помощью функции model_catboost()")
            self.model = None
            self.is_loaded = False

    def predict_batch(self, X_data: pd.DataFrame = None):
        # Предсказание для набора данных
        if not self.is_loaded:
            print("Модель не загружена")
            return None

        if X_data is None:
            print("Не переданы данные для предсказания")
            return None

        # Получение вероятности и предсказания
        probabilities = self.model.predict_proba(X_data)[:, 1]
        predictions = self.model.predict(X_data)

        # Формирование результата
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'employee_id': X_data.index[i] if hasattr(X_data.index, '__getitem__') else i,
                'prediction': int(pred),
                'probability': float(prob),
                'will_leave': bool(pred == 1)
            })

        return results

    def predict_single(self, employee_data: Dict):
        # Предсказание для одного сотрудника
        if not self.is_loaded:
            print("Модель не загружена")
            return None

        # Преобразуем словарь в DataFrame
        df = pd.DataFrame([employee_data])

        # Предсказание
        probability = self.model.predict_proba(df)[0, 1]
        prediction = self.model.predict(df)[0]

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'will_leave': bool(prediction == 1)
        }

    def reset_training(self) -> None:
        # Сброс обучения модели для возможности переобучения
        self._is_trained = False
        self._model = None

class PredictionResult (SQLModel, table=True):
    __tablename__ = "prediction_result"
    __table_args__ = {'extend_existing': True}
    prediction_id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.user_id")
    prediction_rez: str # JSON строка
    cost: float
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Связь с пользователем
    #user: Optional["User"] = Relationship(back_populates="predictions")

    def get_prediction_rez(self) -> List[Dict]:
        # Преобразует JSON строку из БД в список словарей Python
        return json.loads(self.prediction_rez)

    def set_prediction_rez(self, data: List[Dict]):
        # Преобразует список словарей Python в JSON строку для сохранения в БД
        self.prediction_rez = json.dumps(data)


class MLModel:
    def __init__(self):
        self._model = None
        self._is_trained = False

    def get_cost_predict(self) -> float:
        #Cтоимость предсказания
        return 10.0

class MLModelService:
    def __init__(self, balance_service: BalanceService):
        # Инициализация модели Iris
        self.imodel = MLModel()
        self.balance_service = balance_service
        #self.predictions = (Predictions(self.imodel, balance))

    def make_prediction(self, user: User, data: List[Dict]) -> List[Dict]:
        cost = self.imodel.get_cost_predict()

        # Используем методы объекта balance_service
        balance = self.balance_service.get_balance(user)
        if balance < cost:
            raise ValueError("Not enough credits")

        predictions = self.imodel.predict(data)

        # Списание средств (нужно передать session)
        # self.balance_obj.update_balance(-cost, session)
        success = self.balance_service.withdraw(user, cost)
        if not success:
            raise ValueError("Withdrawal failed")
        return predictions

        #prediction_result = self.predictions.make_predict(user, data)
        #return prediction_result.get_prediction_rez()

    def reset_model(self) -> None:
        # Сброс модели для переобучения
        self.ml_model.reset_training()