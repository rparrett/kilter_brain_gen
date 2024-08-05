module_dir := "src"

python3 := if os() == 'windows' {
  "venv/Scripts/python"
} else {
  "venv/bin/python3"
}
os_python := if os() == 'windows' {
  "python"
} else {
  "python3"
}

export FLASK_APP := "src/api/api.py"

@default: lint

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

@sync-deps:
  test -f requirements.txt && \
  ({{python3}} -m pip install --upgrade pip pip-tools && \
  {{python3}} -m piptools sync)

@update-deps:
  ({{python3}} -m pip install --upgrade pip pip-tools && \
  {{python3}} -m piptools compile --resolver=backtracking && {{python3}} -m pip install -r requirements.txt)

@mypy:
  {{python3}} -m mypy --pretty {{module_dir}}/

@isort:
  {{python3}} -m isort --float-to-top {{module_dir}}/

@venv:
  test -f {{python3}} || {{os_python}} -m venv venv

@run *options: venv deps
  {{python3}} {{options}}

@tensorboard model: venv deps
  venv/bin/tensorboard --logdir={{model}}-model

@flask *options: venv deps
  venv/bin/flask run --host=0.0.0.0 --port=5001
