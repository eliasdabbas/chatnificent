---
name: example-app
description: Create an example Chatnificent app to be showcased in the /examples directory.
---

# Example App Development

Use this skill when creating a new example app for the `/examples/` directory.

All examples live in `/examples/` as standalone scripts. See `examples/README.md` for the full index. Review existing examples before writing a new one.

## Constraints

- Each example is a **single `.py` file**, not a directory
- **PEP 723** `# /// script` metadata block declares dependencies — use `uv add --script` to set this up
- Every example has an `if __name__ == "__main__": app.run()` guard
- Minimal code — focus on showcasing Chatnificent's API, not unrelated logic
- Examples should demonstrate user-facing features that make Chatnificent a delight to develop with
- Avoid using `print()` to demonstrate functionality

## File Structure

Each example follows this order:

1. `# /// script` metadata block (PEP 723 inline dependencies)
2. Triple-quoted module docstring
3. App code
4. `if __name__ == "__main__":` guard

## Module Docstring

The docstring is the most important part. It should be a **self-contained how-to guide** that explains:

- What the example demonstrates and why it's useful
- How to run it (`uv run examples/<name>.py`)
- What Chatnificent features/pillars are used
- What to explore next (related examples, customization ideas)

Be extensive and specific. A reader should understand the "why" behind every configuration choice, not just the "what."
