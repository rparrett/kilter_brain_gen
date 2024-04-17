module_dir := "src"
python3 := "venv/bin/python3"

export FLASK_APP := "src/api/api.py"

@default: test lint

@lint:
  {{python3}} -m black {{module_dir}}/ --check
  {{python3}} -m ruff check {{module_dir}}/

@fmt:
  {{python3}} -m ruff format {{module_dir}}/
  {{python3}} -m isort --profile black --float-to-top {{module_dir}}/

@vulture:
	{{python3}} -m vulture {{module_dir}}/ --min-confidence 80

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
  venv/bin/tensorboard --logdir=name-model

@flask *options: venv deps
  venv/bin/flask run --host=0.0.0.0 --port=5001
