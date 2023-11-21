**chrono-kit** is an open source python library for time series analysis and forecasting.

This project was started in 2023 by ODTU YZT.
## Documentation
See the [wiki](https://github.com/odtuyzt/chrono-kit/wiki) for documentation.

Or take a look at the [examples](https://github.com/odtuyzt/chrono-kit/tree/main/examples).

## Installation

### Dependencies

chronokit requires:

* Poetry

After having Poetry installed, under chrono-kit folder you can specify virutal environment path for Poetry:

```bash
poetry config virtualenvs.in-project true
```

Then you can create new virtual environemnt under existing project by running the command below, which will create `.venv`

```bash
poetry install --no-root
```

Note:
If you do not provide --no-root it will try to install project package too.

Now you can activate environment:

```bash
poetry shell
```

Note:
To add new library with specific version you can run the command below (for example library demo >= 1.1.5)

```bash
poetry add demo=^1.1.5
```


### User Installation

    pip install chrono-kit
