# CLAUDE.md - AI Assistant Guide for Polytrader

This document provides guidance for AI assistants working with the Polytrader codebase.

## Project Overview

**Polytrader** is a trading-related project. This repository is currently in its initial setup phase.

**Repository Status:** Newly initialized - awaiting initial codebase setup.

## Repository Structure

```
polytrader/
├── CLAUDE.md          # This file - AI assistant guidance
└── .git/              # Git repository
```

*This section should be updated as the project structure develops.*

## Development Setup

### Prerequisites

*To be documented as dependencies are added.*

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd polytrader

# Install dependencies (once package manager is configured)
# npm install / yarn install / pip install -r requirements.txt
```

### Environment Configuration

*Document any required environment variables here once they are established.*

## Development Workflow

### Branch Naming Convention

- Feature branches: `feature/<description>`
- Bug fixes: `fix/<description>`
- Claude AI branches: `claude/<session-id>`

### Git Practices

1. Create descriptive commit messages
2. Keep commits atomic and focused
3. Push to feature branches, not directly to main
4. Create pull requests for code review

### Build Commands

*To be documented once build system is configured.*

```bash
# Placeholder commands - update as project develops
# npm run build
# npm run dev
# npm run test
# npm run lint
```

## Code Conventions

### General Guidelines

- Write clean, readable, and maintainable code
- Follow the principle of least surprise
- Document complex logic with comments
- Keep functions focused and single-purpose

### Style Guide

*Update this section with specific style guides once the tech stack is established:*
- For TypeScript/JavaScript: ESLint + Prettier configuration
- For Python: PEP 8, Black formatter
- For other languages: Document conventions as needed

## Architecture

*This section should document:*
- System architecture diagrams
- Key design patterns used
- Module responsibilities
- Data flow descriptions

## Testing

### Test Strategy

*Document testing approach:*
- Unit tests
- Integration tests
- End-to-end tests

### Running Tests

```bash
# Placeholder - update with actual commands
# npm test
# pytest
```

## Key Files Reference

*Update this section with important files as they are created:*

| File | Purpose |
|------|---------|
| `CLAUDE.md` | AI assistant guidance document |

## Common Tasks

### For AI Assistants

When working on this codebase:

1. **Before making changes:** Read relevant files to understand context
2. **Making edits:** Use targeted, minimal changes
3. **After changes:** Run tests and linting if available
4. **Committing:** Write clear, descriptive commit messages

### Troubleshooting

*Document common issues and solutions here.*

## Security Considerations

- Never commit secrets, API keys, or credentials
- Use environment variables for sensitive configuration
- Review code for security vulnerabilities (OWASP Top 10)

## API Documentation

*Link to or document API endpoints as they are created.*

## Dependencies

*List major dependencies and their purposes once added.*

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-01-23 | Initial CLAUDE.md creation | Claude AI |

---

*This document should be updated as the project evolves. Add sections for specific frameworks, tools, and conventions as they are adopted.*
