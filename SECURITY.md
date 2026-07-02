# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.4.x   | :white_check_mark: |
| < 1.4   | :x:                |

## Reporting a Vulnerability

Please report security issues by opening a **private issue** on GitHub (if available) or emailing the maintainers directly. **Do not open public issues for vulnerabilities.**

Include:
- Affected version
- Description of the vulnerability
- Steps to reproduce (if applicable)
- Suggested fix (if any)

You should receive a response within 72 hours.

## Security Model — `run` Command

The `run` command allows users to execute arbitrary Python scripts. It is designed as a **trusted-user feature** — similar to公式编辑 in JMP or Excel. It is **NOT** designed to run untrusted code from external sources.

### Threat model

- ✅ Intended: power users transforming their own data with simple scripts
- ❌ Not intended: executing code from untrusted third parties, multi-tenant environments

### Defenses in place

1. **AST whitelist parsing** — only explicitly allowed AST node types pass validation
2. **Import ban** — `import` and `from ... import` statements are rejected
3. **Dunder access blocked** — any attribute or name starting with `__` is rejected (prevents escape via `__class__`, `__bases__`, `__subclasses__`, etc.)
4. **Restricted builtins** — `eval`, `exec`, `compile`, `print`, `open`, `__import__` are removed from the execution namespace
5. **Timeout** — scripts are killed after a configurable timeout (default 5s) via `ctypes.pythonapi.PyThreadState_SetAsyncExc`
6. **Size limit** — scripts are capped at 100 KB
7. **No file system access** — no `open()`, `os`, `shutil`, or similar available in the sandbox

### Known limitations

- The `ctypes`-based timeout uses CPython internals and is not guaranteed to interrupt all operations (e.g., C-extension compute loops)
- The sandbox is **not** a container or VM isolation — a sophisticated attacker with knowledge of CPython internals may find bypass routes
- `run` is disabled by default in CI environments and intended for local/interactive use only

## Input Validation Approach

All user inputs go through layered validation:

1. **Structural validation** — JSON schema checked at the handler level
2. **Size limits** — arrays capped at 100,000 elements (prevents OOM)
3. **Type coercion** — numeric inputs validated and converted explicitly
4. **File handling** — file paths validated; loaded through sanitized loaders (pandas/openpyxl)
5. **Error classification** — ValueError messages classified into error types (PARAM_ERROR, DATA_ERROR, COMPUTATION_ERROR) before returning to the caller

### Input size limits

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Max size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>values</code></td>
      <td>100,000 elements</td>
    </tr>
    <tr>
      <td><code>groups</code> (total)</td>
      <td>100,000 elements</td>
    </tr>
    <tr>
      <td><code>x</code>, <code>y</code></td>
      <td>100,000 elements each</td>
    </tr>
    <tr>
      <td><code>script</code> (run command)</td>
      <td>100 KB</td>
    </tr>
  </tbody>
</table>

## Dependency Security

Dependencies are pinned to major versions in `pyproject.toml`. Security updates to dependencies are picked up through patch releases. Run `pip audit` to check for known CVEs in the dependency tree.
