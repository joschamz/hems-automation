# Security Policy

## Supported Scope

This repository is maintained as a capstone/research project. Security fixes are handled on a best-effort basis.

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Report suspected vulnerabilities privately to the maintainers through GitHub private reporting (Security tab) or direct maintainer contact.

Include:

- A clear description of the issue
- Steps to reproduce
- Potential impact
- Suggested remediation (if known)

## Secret Handling

- Never commit real credentials or API keys.
- Store local secrets only in ignored files under `secrets/`.
- If a secret is exposed, rotate/revoke it immediately.
