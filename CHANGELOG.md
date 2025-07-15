# CHANGELOG

All notable changes to the `mlflow-assistant` package will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and follows changelog conventions inspired by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [v0.1.4] – 2025-07-15

### Added
- **Setup** This allows users to setup mlflow-assistant.
- **Start** This allows users to start the session to interact with mlflow interactively.

### Notes
- Start functionality is currently limited as it is calling a mock function

## [v0.1.3] – 2025-07-15

### Fixed

- Fixed navigation structure in `mkdocs.yml` to correctly point to `reference/` directory instead of `reference/SUMMARY.md`
- Improved compatibility with `literate-nav` and `section-index` plugins for proper code reference navigation

## [v0.1.2] – 2025-07-15

### Added

- **Comprehensive documentation system** with MkDocs Material theme
- **Versioned documentation deployment** using mike for GitHub Pages
- **Automated documentation generation** from code docstrings using mkdocstrings
- **Code reference navigation** with auto-generated API documentation
- **Custom Geist font** for modern, clean documentation appearance
- **Integrated documentation deployment** in the main CI/CD pipeline

### Changed

- Enhanced CI/CD pipeline to include documentation deployment after successful releases
- Documentation now automatically deploys on version bumps and PyPI releases

### Notes

- Documentation is available at <https://hugodscarvalho.github.io/mlflow-assistant/>
- Each release gets its own versioned documentation with `latest` alias pointing to the newest version

## [v0.1.1] – 2025-07-14

### Added

- Introduced a **unified CI/CD pipeline** for:
  - Linting with `ruff`.
  - Type checking with `mypy`.
  - Test execution with `pytest` and coverage reporting via `pytest-cov`.
  - Publishing to PyPI on new Git tags.

### Changed

- Improved test fixture logic for integration tests to correctly check for a live MLflow Tracking Server before execution.

### Notes

- This pipeline simplifies maintenance by consolidating quality checks and publishing into a single workflow, triggered on pushes and releases.

## [v0.1.0] – 2025-05-08

### Added

- **Initial placeholder release** of `mlflow-assistant` on PyPI.
- Included a minimal `MLflowClient` wrapper module.

### Notes

- This version was intentionally published with limited functionality to reserve the `mlflow-assistant` package name on [PyPI](https://pypi.org/project/mlflow-assistant/).
- Full development will begin in upcoming versions with a focus on making MLflow easier to use through high-level utilities and assistant-like automation features.
