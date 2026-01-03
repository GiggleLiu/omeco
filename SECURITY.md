# Security Policy

## Supported Versions

We currently support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in omeco, please report it by:

1. **DO NOT** open a public issue
2. Email the maintainers directly or use GitHub's private vulnerability reporting feature
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge your report within 48 hours and provide a detailed response within 7 days, including:
- Confirmation of the vulnerability
- Our planned timeline for a fix
- Any workarounds available

## Security Best Practices

When using omeco:

- Keep dependencies up to date
- Review the dependency tree for known vulnerabilities using `cargo audit`
- Follow Rust security best practices
- Validate all input data before passing to optimization functions

## Disclosure Policy

We follow coordinated vulnerability disclosure:

1. Vulnerability is reported privately
2. Fix is developed and tested
3. Security advisory is published
4. Fixed version is released
5. Public disclosure after users have time to update

Thank you for helping keep omeco and its users safe!

