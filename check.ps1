isort .
black .

mypy src
pylint src

coverage run -m unittest discover ./tests/unit
coverage combine --quiet
coverage report
