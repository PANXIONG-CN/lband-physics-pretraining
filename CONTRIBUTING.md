# Contributing

Contributions should keep the numerical protocol and data provenance visible.

1. Create a focused branch.
2. Do not commit raw third-party data, credentials, local absolute paths, or
   generated experiment directories.
3. Add or update a contract test for every reader, physical model, or metric
   change.
4. Run the unit tests and standalone checks listed in `README.md`.
5. Report changes to sample definitions, random seeds, data splits, units, or
   aggregation rules in the pull request.

New experiments must write to a new versioned output directory. Existing
frozen results must not be overwritten.

Scientific changes should distinguish exploratory analysis from locked or
pre-registered evaluation. Results observed in the target campaign must not be
used to relabel an experiment as prospective.
