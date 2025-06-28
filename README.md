# kilter_brain_gen

Generating kilter board problems with transformers

## Usage

### Clone the repo

```bash
git clone https://github.com/rparrett/kilter_brain_gen
cd kilter_brain_gen
```

### Install dependencies

We're using [uv](https://docs.astral.sh/uv/getting-started/installation/) for package and project management.

- Run `uv sync`
- (Windows / CUDA) (TODO, this might be configured correctly in `pyproject.toml`)

### Get climb data

- Install [`sqlite`](https://www.sqlite.org/download.html)
- Use [boardlib](https://github.com/lemeryfertitta/BoardLib), or extract the database from a kilter `apk` file, or ask someone who has it.
- Move that database to `data/climbs.sqlite3`
- Run `uv run src/data/get_csv.py`

### Train the `climb_clm` model

`uv run src/climb_clm/train.py`

### Generate some climbs

`uv run src/climb_clm/generate.py`

### Run the API server

- `uv run flask -A src/api/api.py run`
- Test with `curl -X POST -H "Content-Type: application/json" -d '{"prompt":""}' 'http://127.0.0.1:5000/generate'`
- Debug builds of [`kilter_brain`](https://github.com/rparrett/kilter_brain) will connect to the local server.

## Development

### Linting

`uv run ruff check`

### Formatting

`uv run ruff format`

### Monitoring Experiments

`uv run tensorboard --logdir models`

## TODO

- [ ] Train a model for route names that actually works
- [ ] Tidy everything up with a nice CLI framework
- [ ] Add similarity search
- [ ] Somehow specify windows-specific dependencies that grab torch with CUDA support

### Experiments

- [ ] Try adding duplicate climbs with randomized frame data ([shuffle-frames](https://github.com/rparrett/kilter_brain_gen/tree/shuffle-frames))
- [ ] Try different sampling strategies for climb data
