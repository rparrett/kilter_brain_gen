# kilter_brain_gen

Generating kilter board problems with transformers

## Usage

### Get the kilter sqlite database

- Install [`sqlite`](https://www.sqlite.org/download.html)
- Use [boardlib](https://github.com/lemeryfertitta/BoardLib) or extract from a kilter `apk` file.
- Run `get_csv.sh`

### Install dependencies

- Install [just](https://github.com/casey/just?tab=readme-ov-file#installation)
- Run `just sync-deps`

### Train the `climb` model

- `just run src/climb_clm/train_tokenizer.py`
- `just run src/climb_clm/train.py`

### Generate some climbs

- `just run src/climb_clm/generate.py`

### Run the API server

- `just flask`
- Debug builds of [`kilter_brain`](https://github.com/rparrett/kilter_brain) will connect to the local server.

## TODO

- [ ] Train a model for route names that actually works
- [ ] Tidy everything up with a nice CLI framework
- [ ] Add similarity search
- [ ] Somehow specify windows-specific dependencies that grab torch with CUDA support

### Experiments

- [ ] Try adding duplicate climbs with randomized frame data ([shuffle-frames](https://github.com/rparrett/kilter_brain_gen/tree/shuffle-frames))
- [ ] Try different sampling strategies for climb data
