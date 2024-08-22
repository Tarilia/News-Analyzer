lint: # запуск линтера
	poetry run flake8 news_analyzer

install: # установка пакета после клонирования репозитория или удаления зависимостей
	poetry install

run: # запуск
	poetry run run
