import subprocess
import sys


def run_tests():
    # Тестовые файлы
    test_files = [
        "appp/test/test_user.py",
        "appp/test/test_model.py",
        "appp/test/test_recommendations.py"
    ]

    for test_file in test_files:

        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file, "-v"
        ], capture_output=True, text=True)

        print(result.stdout)
        if result.stderr:
            print("Ошибки:")
            print(result.stderr)

if __name__ == "__main__":
    run_tests()

# Запуск всех тестов
#pytest appp/test/test_auth.py
