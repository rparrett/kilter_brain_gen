## Experiments

When experimenting with hyperparameters during training, you may want to pass a specific run name to the script to identify the outputs. For example, `just train --run-name larger-batch-size`.
By default, the run name is generated sequentially as `run-000`.

## Tensorboard

To start tensorboard to view data for all training runs, use `just run tensorboard` which defaults to finding logs for runs recursively in the `training/` directory.
For a specific run, use `just run tensorboard run-000`.

Runs are stored in `training/` directory, with each subdirectory corresponding to a specific unique run.

For example:

```
training/run-000/
training/run-000/events.out.tfevents.1587350000.name
training/run-001/
training/run-001/events.out.tfevents.1587350123.name 
training/try-larger-batch-size/
training/try-larger-batch-size/events.out.tfevents.1587350123.name
```

See <https://github.com/tensorflow/tensorboard/blob/master/README.md> for info about the recommended organization of logfiles.
