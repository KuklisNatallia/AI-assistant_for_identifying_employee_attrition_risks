from fastapi import FastAPI, Request, Depends, HTTPException, status, Response
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
import secrets
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select
import bcrypt
from datetime import datetime
from typing import Optional, List
import json
import sys
import os
import pandas as pd
import joblib
import logging

# Очищаем метаданные SQLAlchemy перед импортом моделей
from sqlalchemy import MetaData
metadata = MetaData()
metadata.clear()

# Добавляем корневую директорию проекта в путь Python
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Поднимаемся на уровень выше
sys.path.insert(0, project_root)
sys.path.insert(0, current_dir)

from appp.models.models import PredictionResult, TrainedModel
from appp.database.databases import engine
from appp.database.config import Settings, get_settings
from appp.models.models import PredictionResult, MLModel, MLModelService, model_catboost
from appp.models.balance import BalanceService as ImportedBalanceService
from appp.models.user import User, get_user_by_email, get_user_by_id, create_user
from appp.models.balance import Balance
from appp.services.jwt_handler import create_access_token, verify_access_token, get_current_user_from_token
from appp.services.cookieauth import OAuth2PasswordBearerWithCookie
from appp.services.loginform import LoginForm
from appp.models.transaction import TransactionType, Trans
from appp.services.analyze import analyze_at_risk
from appp.services.shap import shap_analyz
from appp.services.hr_recommend import RecommendSystem
from appp.services.simpleLLM import SimpleOllamaEngine
from appp.services.judge import SimpleJudge


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=secrets.token_urlsafe(32),  # Секретный ключ для сессий
    session_cookie="session",
    max_age=3600
)
settings = get_settings()

# Настройка шаблонов и статических файлов
templates = Jinja2Templates(directory="templates")
# appp.mount("/static", StaticFiles(directory="static"), name="static")

# OAuth2 схема с куками
oauth2_scheme = OAuth2PasswordBearerWithCookie(
   tokenUrl="/api/users/login",
   auto_error=False
)

# Глобальные переменные для модели и данных
trained_model = None
X_test_data = None
model_loaded = False
trained_model_instance = None

def load_trained_model():
    # Загружаем обученную модель
    global trained_model_instance, X_test_data, model_loaded
    logger.info("Загрузка модели...")

    # Создаем экземпляр TrainedModel
    trained_model_instance = TrainedModel(model_path="catboost_model.joblib")

    # Если модель не загружена, обучаем ее
    if not trained_model_instance.is_loaded:
        logger.info("Обучаем модель...")
        model_results = model_catboost(save_model=True, model_path="catboost_model.joblib")
        trained_model_instance.model = model_results['model']
        trained_model_instance.is_loaded = True
        X_test_data = model_results['X_test']
        logger.info("Модель обучена и сохранена")
    else:
        # Убеждаемся что rained_model_instance.model не None
        if trained_model_instance.model is None:
            logger.warning("Модель загружена из файла, но объект модели не инициализирован")
            # Перезагрузим модель из файла
            trained_model_instance.model = joblib.load("catboost_model.joblib")

        # Нужно получить X_test из данных
        from appp.models.models import preprocessing
        df = preprocessing()
        X = df.drop('Attrition', axis=1)
        from sklearn.model_selection import train_test_split
        _, X_test, _, _ = train_test_split(X, df['Attrition'],
                                           test_size=0.2,
                                           random_state=42,
                                           stratify=df['Attrition'])
        X_test_data = X_test

    logger.info(f"Модель загружена. X_test_data размер: {X_test_data.shape}")


class BalanceService:
    def __init__(self, session: Session):
        self.session = session

    def get_balance(self, user: User) -> float:
        balance = self.session.get(Balance, user.user_id)
        if not balance:
            # Создаем баланс если его нет
            balance = Balance(user_id=user.user_id, amount=0.0)
            self.session.add(balance)
            self.session.commit()
            self.session.refresh(balance)
        return balance.amount

    def deposit(self, user: User, amount: float) -> None:
        balance = self.session.get(Balance, user.user_id)
        if not balance:
            balance = Balance(user_id=user.user_id, amount=amount)
        else:
            balance.amount += amount

        self.session.add(balance)

        # Создаем запись о транзакции
        transaction = Trans(
            transaction_id=f"trans_{datetime.now().timestamp()}",
            user_id=user.user_id,
            amount=amount,
            transaction_type=TransactionType.DEPOSIT.value
        )
        self.session.add(transaction)
        self.session.commit()

    def withdraw(self, user: User, amount: float) -> bool:
        balance = self.session.get(Balance, user.user_id)
        if not balance or balance.amount < amount:
            return False

        balance.amount -= amount
        self.session.add(balance)

        # Создаем запись о транзакции
        transaction = Trans(
            transaction_id=f"trans_{datetime.now().timestamp()}",
            user_id=user.user_id,
            amount=-amount,
            transaction_type=TransactionType.COST_PREDICTION.value
        )
        self.session.add(transaction)
        self.session.commit()
        return True


# получение текущего пользователя
async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    if not token:
        return None

    try:
        user_data = get_current_user_from_token(token)
        with Session(engine) as db_session:
            user = db_session.get(User, user_data["user_id"])
            return user
    except HTTPException:
        return None


# аутентифицированные пользователи
async def get_authenticated_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )

    user_data = get_current_user_from_token(token)
    with Session(engine) as db_session:
        user = db_session.get(User, user_data["user_id"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        return user


@app.get("/", response_class=HTMLResponse)
async def index(
        request: Request,
        current_user: Optional[User] = Depends(get_current_user)
):
    error_message = request.query_params.get("error")
    success_message = request.query_params.get("success")

    if current_user:
        with Session(engine) as db_session:
            balance_service = BalanceService(db_session)
            balance = balance_service.get_balance(current_user)

            # Проверяем наличие последнего предсказания без сохранения в БД
            prediction_result = request.session.get("prediction_results", [])
            prediction_made = len(prediction_result) > 0

            return templates.TemplateResponse("app.html", {
                "request": request,
                "registered": True,
                "username": current_user.username,
                "balance": balance,
                "prediction_made": prediction_made,
                "prediction_result": prediction_result,
                "error_message": error_message,
                "success_message": success_message
            })

    return templates.TemplateResponse("app.html", {
        "request": request,
        "registered": False,
        "error_message": error_message,
        "success_message": success_message
    })


@app.post("/api/users/register")
async def register(request: Request, response: Response):
    form_data = await request.form()
    username = form_data.get("username", "").strip()
    email = form_data.get("email", "").strip()
    password = form_data.get("password", "").strip()

    # Валидация
    if not all([username, email, password]):
        return RedirectResponse("/?error=Все поля обязательны для заполнения", status_code=303)

    if len(password) < 8:
        return RedirectResponse("/?error=Пароль должен содержать минимум 8 символов", status_code=303)

    if "@" not in email or "." not in email:
        return RedirectResponse("/?error=Некорректный формат email", status_code=303)

    with Session(engine) as db_session:
        # Проверяем, нет ли уже такого пользователя
        existing_user = db_session.exec(select(User).where(User.email == email)).first()
        if existing_user:
            return RedirectResponse("/?error=Пользователь с таким email уже существует", status_code=303)

        # Создаем нового пользователя
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        new_user = User(
            username=username,
            email=email,
            password_hash=hashed_password,
            created_at=datetime.now(),
            balance=0.0
        )

        try:
            db_session.add(new_user)
            db_session.commit()
            db_session.refresh(new_user)

            # Создаем баланс для пользователя
            balance = Balance(user_id=new_user.user_id, amount=0.0)
            db_session.add(balance)
            db_session.commit()

            # Создаем JWT токен
            token = create_access_token(new_user.user_id, new_user.username)

            # Устанавливаем токен в куки
            response = RedirectResponse("/?success=Регистрация завершена успешно!", status_code=303)
            response.set_cookie(
                key=settings.COOKIE_NAME,
                value=token,
                httponly=True,
                max_age=3600,
                secure=not settings.DEBUG,
                samesite="lax"
            )

            return response

        except Exception as e:
            db_session.rollback()
            return RedirectResponse(f"/?error=Ошибка при создании пользователя: {str(e)}", status_code=303)


@app.post("/api/users/login")
async def login(request: Request, response: Response):
    try:
        form_data = await request.form()
        email = form_data.get("username", "").strip()
        password = form_data.get("password", "").strip()

        if not email or not password:
            return RedirectResponse("/?error=Email и пароль обязательны", status_code=303)

        with Session(engine) as db_session:
            user = db_session.exec(select(User).where(User.email == email)).first()
            if not user:
                return RedirectResponse("/?error=Неверный email или пароль", status_code=303)

            # Проверяем пароль
            try:
                if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                    return RedirectResponse("/?error=Неверный email или пароль", status_code=303)
            except Exception as e:
                return RedirectResponse("/?error=Ошибка проверки пароля", status_code=303)

            # Создаем JWT токен
            token = create_access_token(user.user_id, user.username)

            # Устанавливаем токен в куки
            response = RedirectResponse("/?success=Вход выполнен успешно!", status_code=303)
            response.set_cookie(
                key=settings.COOKIE_NAME,
                value=token,
                httponly=True,
                max_age=3600,
                secure=not settings.DEBUG,
                samesite="lax"
            )

            return response

    except Exception as e:
        print(f"Login error: {e}")
        return RedirectResponse(f"/?error=Ошибка сервера: {str(e)}", status_code=303)

@app.post("/balance")
async def handle_balance(
        request: Request,
        current_user: User = Depends(get_authenticated_user)
):
    form_data = await request.form()
    try:
        amount = float(form_data.get("amount", 10))
    except ValueError:
        return RedirectResponse("/?error=Неверная сумма", status_code=303)

    with Session(engine) as db_session:

        balance_service = BalanceService(db_session)
        balance_service.deposit(current_user, amount)

    return RedirectResponse("/?success=Баланс пополнен успешно!", status_code=303)


@app.post("/prediction")
async def handle_prediction(
        request: Request,
        current_user: User = Depends(get_authenticated_user)):

    global trained_model_instance, X_test_data  # , model_loaded

    # Проверка и загрузка модели
    if trained_model_instance is None:
        load_trained_model()

    if (not trained_model_instance or
            not trained_model_instance.is_loaded or
            X_test_data is None or
            X_test_data.empty):
        # return RedirectResponse("/?error=Модель не загружена", status_code=303)
        # Получаем баланс перед возвратом ошибки
        with Session(engine) as db_session:
            balance_service = BalanceService(db_session)
            balance = balance_service.get_balance(current_user)

        return templates.TemplateResponse("app.html", {
            "request": request,
            "registered": True,
            "username": current_user.username,
            "balance": balance,
            "prediction_made": False,
            "prediction_result": [],
            "error_message": "Модель не загружена"
        })

    if trained_model_instance.model is None:
        # return RedirectResponse("/?error=Внутренняя ошибка модели", status_code=303)
        # Получаем баланс перед возвратом ошибки
        with Session(engine) as db_session:
            balance_service = BalanceService(db_session)
            balance = balance_service.get_balance(current_user)

        return templates.TemplateResponse("app.html", {
            "request": request,
            "registered": True,
            "username": current_user.username,
            "balance": balance,
            "prediction_made": False,
            "prediction_result": [],
            "error_message": "Внутренняя ошибка модели"
        })

    # Обработка баланса и списание стоимости предсказания
    with Session(engine) as db_session:
        balance_service = BalanceService(db_session)
        balance = balance_service.get_balance(current_user)

        cost = 10.0  # стоимость услуги

        if balance < cost:
            return RedirectResponse("/?error=Недостаточно средств", status_code=303)


        success_withdraw = balance_service.withdraw(current_user, cost)
        if not success_withdraw:
            # return RedirectResponse("/?error=Недостаточно средств", status_code=303)
            return templates.TemplateResponse("app.html", {
                "request": request,
                "registered": True,
                "username": current_user.username,
                "balance": balance,
                "prediction_made": False,
                "prediction_result": [],
                "error_message": "Недостаточно средств"
            })

    # Анализ с моделью
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Создаем ThreadPoolExecutor для выполнения CPU-интенсивной задачи
        executor = ThreadPoolExecutor(max_workers=1)

        # Запускаем анализ в отдельном потоке
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            executor,
            lambda: analyze_at_risk(trained_model_instance.model, X_test_data, threshold=0.9)
        )
        # model = trained_model_instance.model
        # Запускаем анализ с порогом 0.9 (90%)
        # results = analyze_at_risk(model, X_test_data, threshold=0.9) # 0.7 (70%)

        logger.debug(f"Результатов анализа: {len(results)}")

        # Проверяем и форматируем результаты
        session_results = []
        for i, result in enumerate(results):
            if isinstance(result, dict):
                # Копируем результат
                formatted_result = result.copy()
                # Проверяем наличие ID
                if 'employee_id' not in formatted_result:
                    formatted_result['employee_id'] = f"emp_{i + 1}"

                # Форматируем вероятность
                if 'probability' in formatted_result and 'probability_percent' not in formatted_result:
                    prob = float(formatted_result['probability'])
                    formatted_result['probability_percent'] = f"{prob * 100:.1f}%"

                # Убеждаемся, что рекомендации - это список
                if 'recommendations' in formatted_result and isinstance(formatted_result['recommendations'], str):
                    # Разделяем строку на список рекомендаций
                    recs = [r.strip() for r in formatted_result['recommendations'].split('\n') if r.strip()]
                    formatted_result['recommendations'] = recs

                elif 'recommendations' not in formatted_result:
                    formatted_result['recommendations'] = []

                session_results.append(formatted_result)
            else:
                # Если результат не словарь, создаем базовую структуру
                session_results.append({
                    'employee_id': f"emp_{i + 1}",
                    'probability': 0.0,
                    'probability_percent': '0.0%',
                    'original_data': str(result)[:200],
                    'recommendations': ['Не удалось получить рекомендации'],
                    'messages': [f'Исходные данные: {str(result)[:100]}...']
                })

        # Сохраняем в сессии
        # request.session["prediction_results"] = session_results
        # request.session["prediction_time"] = datetime.now().isoformat()
        # request.session["total_employees"] = len(session_results)

        # Явно сохраняем сессию
        #await request.session._save()
        request.session.update({
            "prediction_results": session_results,
            "prediction_time": datetime.now().isoformat(),
            "total_employees": len(session_results),
            "prediction_saved": True
        })

        # Сохраняем сессию как измененную
        # request.session.setdefault("__modified__", True)

        # Получаем обновленный баланс после списания
        with Session(engine) as db_session:
            balance_service = BalanceService(db_session)
            updated_balance = balance_service.get_balance(current_user)

        logger.info(f"Сохранено {len(session_results)} записей в сессию")

        # Перенаправляем на страницу результатов
        # return RedirectResponse(
        #     "/results?success=Предсказание выполнено успешно!",
        #     status_code=303
        # )
        # Возвращаем HTML-ответ с данными
        return templates.TemplateResponse("app.html", {
            "request": request,
            "registered": True,
            "username": current_user.username,
            "balance": updated_balance,
            "prediction_made": len(session_results) > 0,
            "prediction_result": session_results,
            "total_employees": len(session_results),
            "success_message": f"Анализ выполнен успешно! Найдено {len(session_results)} сотрудников"
        })

    except Exception as e:
        logger.error(f"ERROR в handle_prediction: {e}")
        import traceback
        traceback.print_exc()
        #return RedirectResponse(f"/?error=Ошибка рекомендаций: {str(e)}", status_code=303)

        # Получаем баланс при ошибке
        with Session(engine) as db_session:
            balance_service = BalanceService(db_session)
            balance = balance_service.get_balance(current_user)

        return templates.TemplateResponse("app.html", {
            "request": request,
            "registered": True,
            "username": current_user.username,
            "balance": balance,
            "prediction_made": False,
            "prediction_result": [],
            "error_message": f"Ошибка анализа: {str(e)[:100]}"
        })


@app.post("/logout")
async def logout(response: Response):
    # Очищаем куки
    response = RedirectResponse("/?success=Выход выполнен успешно!", status_code=303)
    response.delete_cookie(settings.COOKIE_NAME)
    return response


@app.get("/api/predictions")
async def get_user_predictions(
        current_user: User = Depends(get_authenticated_user)
):
    # API для получения предсказаний пользователя
    with Session(engine) as db_session:
        statement = select(PredictionResult).where(
            PredictionResult.user_id == current_user.user_id
        ).order_by(PredictionResult.created_at.desc())

        predictions = db_session.exec(statement).all()
        return {
            "predictions": [
                {
                    "id": pred.prediction_id,
                    "result": pred.get_prediction_rez(),
                    "cost": pred.cost,
                    "created_at": pred.created_at
                } for pred in predictions
            ]
        }


@app.get("/api/user/balance")
async def get_user_balance(
        current_user: User = Depends(get_authenticated_user)
):
    # API для получения баланса пользователя
    with Session(engine) as db_session:
        balance_service = BalanceService(db_session)
        balance = balance_service.get_balance(current_user)
        return {"balance": balance}

# Отображение результатов
@app.get("/results", response_class=HTMLResponse)
async def show_results(
        request: Request,
        current_user: Optional[User] = Depends(get_current_user)):

    # Получаем данные из сессии
    prediction_result = request.session.get("prediction_results", [])

    with Session(engine) as db_session:
        balance_service = BalanceService(db_session)
        balance = balance_service.get_balance(current_user) if current_user else 0

    return templates.TemplateResponse("app.html", {
        "request": request,
        "registered": True if current_user else False,
        "username": current_user.username if current_user else "",
        "balance": balance,
        "prediction_made": len(prediction_result) > 0,
        "prediction_result": prediction_result,
        "total_employees": len(prediction_result),
        "success_message": "Рекомендации выполнены успешно!"
    })


# API для получения данных порциями
@app.get("/api/employees")
async def get_employees(
        request: Request,
        page: int = 1,
        page_size: int = 4
):
    # Получаем данные из сессии
    prediction_result = request.session.get("prediction_results", [])

    if not prediction_result:
        return JSONResponse({
            "employees": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0
        })

    total = len(prediction_result)
    start = (page - 1) * page_size
    end = start + page_size

    # Берем срез данных для текущей страницы
    page_data = prediction_result[start:end]

    # Убедимся, что данные имеют правильную структуру
    formatted_data = []
    for i, emp in enumerate(page_data):
        # Копируем словарь
        formatted_emp = emp.copy()
        
        # Добавляем недостающие поля
        if 'employee_id' not in formatted_emp:
            formatted_emp['employee_id'] = f"emp_{start + i + 1}"

        # Форматируем вероятность
        if 'probability' in formatted_emp and 'probability_percent' not in formatted_emp:
            prob = float(formatted_emp['probability'])
            formatted_emp['probability_percent'] = f"{prob * 100:.1f}%"

        # Убедимся, что рекомендации - это список
        if 'recommendations' in formatted_emp and isinstance(formatted_emp['recommendations'], str):
            formatted_emp['recommendations'] = [r.strip() for r in formatted_emp['recommendations'].split('\n') if r.strip()]
        elif 'recommendations' not in formatted_emp:
            formatted_emp['recommendations'] = []

        formatted_data.append(formatted_emp)
    else:
        # Если данные не в формате словаря, создаем базовую структуру
        formatted_data.append({
            'employee_id': f"emp_{start + i + 1}",
            'probability': 0.0,
            'probability_percent': '0.0%',
            'original_data': str(emp)[:200],
            'recommendations': ['Данные требуют обработки'],
            'messages': ['Некорректный формат данных']
        })

    total_pages = (total + page_size - 1) // page_size

    return JSONResponse({
        "employees": formatted_data,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "showing": f"{start + 1}-{min(end, total)} из {total}"
    })


if __name__ == "__main__":
    import uvicorn

    load_trained_model()

    uvicorn.run(
        app,
        host=settings.app_host,
        port=int(settings.app_port),
        #reload=True,
        log_level="debug"
    )