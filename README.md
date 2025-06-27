# kilter_brain_gen

Generating kilter board problems with transformers

## Usage

### Clone the repo

```bash
git clone https://github.com/rparrett/kilter_brain_gen
cd kilter_brain_gen
```

### Get the kilter sqlite database

- Install [`sqlite`](https://www.sqlite.org/download.html)
- Use [boardlib](https://github.com/lemeryfertitta/BoardLib) or extract from a kilter `apk` file.
- Run `get_csv.sh`

### Install dependencies

We're using [uv](https://docs.astral.sh/uv/getting-started/installation/) for package and project management.

- Run `uv sync`
- (Windows / CUDA) (TODO, this might just work as-is)

### Train the `climb_clm` model

- `uv run src/climb_clm/train_tokenizer.py`
- `uv run src/climb_clm/train.py`

### Generate some climbs

- `uv run src/climb_clm/generate.py`

### Run the API server

- `uv run flask -A src/api/api.py run`
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
