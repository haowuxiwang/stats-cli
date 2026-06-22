# Archive Notice

**This project is archived and no longer actively maintained.**

## Status

- **Version**: 1.2.2 (stable)
- **Archive Date**: 2026-06-22
- **Last Update**: 2026-06-22

## What This Means

- No new features will be added
- No bug fixes will be provided
- No security updates will be released
- Issues and pull requests will not be monitored

## Usage

This project is still functional and can be used as-is. The code has been thoroughly tested and is considered stable for production use.

### Installation

```bash
pip install -r requirements.txt
```

### Testing

```bash
pip install -r requirements-test.txt
pytest tests/
```

## Known Issues

- scipy >= 1.18 changed the `kstest` function signature, which may cause issues with some normality tests. A workaround has been implemented.

## Alternatives

If you need active maintenance, consider:

- [scipy](https://scipy.org/) - Scientific computing library
- [statsmodels](https://www.statsmodels.org/) - Statistical modeling
- [pingouin](https://pingouin-stats.org/) - Statistical package

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions about this project, please contact the repository owner.
