# Contributing

Contributors are welcome! We welcome contributions from the community to make this project even better! Whether it's
code improvements, bug fixes, documentation updates, or new feature suggestions, your input is valuable to us. Feel free
to fork the repository and submit your pull requests.

If you would like to contribute more deeply to NeurIO, please follow how step-by-step guides:

- [Adding a Device to NeurIO](../contribution/ADD_A_DEVICE.md)
- *Adding a Task to NeurIO*: (coming soon)

Make sure to check our [Code of Conduct](./CODE_OF_CONDUCT.md).

## Getting started

If you would like to contribute to NeurIO, then you start by forking the public repositiory, and download it to your
development machine.

```bash
    git clone https://github.com/your_profile/NeurIO
```

Install the package in development mode using pip

```bash
    cd neurio
    pip install -e . --user
```

or

```bash
    pip install -e .[all] --user
```

## Adding a feature

You should commit your modification to a new feature branch.
Pull request will be merged into the``development`` branch.

```bash
    git checkout -b feature/my-feature
    ...
    git commit -m "This is a verbose commit message."
```

Then push your new branch to your repository.

```bash
   git push -u origin feature/my-feature
```

If your modifications aren't already covered by a unit test, please include a unit test with your merge request.
Unit tests go in the ``tests``directory. See the existing tests for examples.
When you are ready, make a merge request on [Github](https://github.com/csem/neurio), from the feature branch of your fork to the development branch of
the original repository.

## Running tests

As part of the merge review process, we will check that all the unit tests pass.
You can run tests locally using the following command:

```bash
    pytest tests
```