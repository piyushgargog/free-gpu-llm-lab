# Security Policy

## About this repository

`free-gpu-llm-lab` is an experimental LLM/GPU benchmarking research lab —
scripts, documentation, and a small shared library for running and
benchmarking open-weight LLMs on free GPU platforms (Google Colab,
Kaggle). It is not a production service, does not run a network-facing
server, and does not handle end-user data. Realistic security concerns
here are mostly about supply-chain/dependency issues and unsafe code
patterns (e.g. command injection in a script), not a hosted attack
surface.

## Supported versions

This is a young, actively-developed research repository with no tagged
releases. There is a single line of development: the `master` branch.
Security fixes are applied there; there are no older maintained versions.

| Branch | Supported |
|---|---|
| `master` | ✅ |

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security vulnerability
you have not yet disclosed.** Public issues are for bugs and feature
requests, not undisclosed security problems.

Instead, use **GitHub's private vulnerability reporting** for this
repository:

1. Go to the [Security tab](https://github.com/piyushgargog/free-gpu-llm-lab/security) of this repository.
2. Click **"Report a vulnerability"**.
3. Fill out the private report form. It goes directly to the repository
   owner ([@piyushgargog](https://github.com/piyushgargog)) and is not
   publicly visible.

If that option isn't visible on your account/client, open a regular issue
titled `[SECURITY]` asking to be pointed to another private contact
channel — do not include exploit details in that issue.

### What to include in a report

A useful report includes:

- A clear description of the vulnerability and its potential impact.
- Steps to reproduce it (or a minimal proof of concept).
- The affected file(s)/script(s) and, if relevant, the exact command or
  configuration that triggers it.
- Whether it requires a GPU/remote environment (Colab/Kaggle) to
  reproduce, or is reproducible from the local-only code paths.
- Your assessment of severity, if you have one — not required.

### What to expect

This is a solo-maintained project without a dedicated security team.
There is no guaranteed response-time SLA, but reports will be read and
acknowledged as soon as reasonably possible, and a fix or mitigation will
be prioritized for anything that looks credible.
