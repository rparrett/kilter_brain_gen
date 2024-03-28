module_dir := "src"

alias t := test
python3 := "venv/bin/python3"

export FLASK_APP := "src/api/api.py"

@default: test lint

@test *options:
	{{python3}} -m pytest -x -vv {{options}} tests/ --cov={{module_dir}} --cov-report=html --cov-report=term

@lint:
  {{python3}} -m black . --check
  {{python3}} -m ruff check .

@fmt:
  {{python3}} -m ruff format {{module_dir}}/ tests/
  {{python3}} -m isort --profile black --float-to-top {{module_dir}}

@ruff:
  {{python3}} -m ruff check --fix {{module_dir}}/ tests/

@vulture:
	{{python3}} -m vulture {{module_dir}}/ tests/ --min-confidence 80

@clean:
	find . -iname "*.pyc" -delete
	find . -iname '__pycache__' -type d | xargs rm -fr
	rm -rf .pytest_cache *.egg-info .out

@deps:
  test -f requirements.txt || \
  ({{python3}} -m pip install --upgrade pip pip-tools && \
  {{python3}} -m piptools compile --resolver=backtracking && {{python3}} -m pip install -r requirements.txt)

@update-deps:
  ({{python3}} -m pip install --upgrade pip pip-tools && \
  {{python3}} -m piptools compile --resolver=backtracking && {{python3}} -m pip install -r requirements.txt)

@mypy:
	{{python3}} -m mypy --pretty {{module_dir}}/

@isort:
  {{python3}} -m isort --float-to-top {{module_dir}}/

@venv:
  test -f venv/bin/python3 || python3 -m venv venv

@run *options: venv deps
  {{python3}} {{options}}

@tensorboard: venv deps
  venv/bin/tensorboard --logdir=climbbert

@flask *options: venv deps
  venv/bin/flask run --host=0.0.0.0 --port=5001 --debug
