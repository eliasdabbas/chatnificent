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
- **Every example ships as a pair: `<feature>_simple.py` and `<feature>_advanced.py`** — see "Two Versions per Example" below

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

## Two Versions per Example

Every example feature ships as **two files**: a *simple* version and an *advanced* version. This is non-negotiable — even a "trivial" feature gets both.

### Why two versions

- **Simple** serves newcomers and the wow-factor: *"I only had to override one method and pass one parameter to get feature X?!"* It is the educational and the inspirational artifact.
- **Advanced** serves real users: it shows the full parameter surface of the feature and how it composes with the rest of the framework, so a developer can lift it into a production app.

### Naming convention

```
examples/<feature>_simple.py
examples/<feature>_advanced.py
```

No other suffixes. The shared `<feature>` prefix is what makes a pair discoverable in directory listings and README tables.

### What "simple" means

- The **smallest possible diff** from the zero-config app that still demonstrates the feature.
- Exactly one override, one subclass, or one parameter — whatever the feature requires, and **nothing else**.
- No unrelated pillars. No auth, no custom store, no extra controls beyond what the feature needs.
- A reader should be able to point at one block of code and say *"that block is the feature."*

### What "advanced" means

- Demonstrates the **parameter / configuration surface** of the feature: multiple options, modes, or knobs the user can choose between.
- Integrates **at least one other pillar** (e.g. a custom Store, Auth, Engine hook, Tools, or Layout subclass) so the example reads as a genuinely useful app, not a toy.
- Reads as something a developer could realistically deploy or fork as a starting point for their own product.
- Still a single `.py` file — "advanced" means richer, not multi-file.

### Canonical reference pair

| Role | File | What it shows |
|------|------|---------------|
| Simple | `examples/ui_interactions.py` | One `Control`, one `DefaultLayout(controls=[...])` call — the entire feature in ~10 lines of app code |
| Advanced | `examples/single_app_multi_chat_mode.py` | Multiple controls + a custom delegating LLM (`ModeRouter`), a custom Engine (`ModeAwareEngine`), and a custom Layout (`MultiChatModeLayout`) — same primitives, production-shaped app |

When in doubt, study this pair: the *simple* file should feel as compact as `ui_interactions.py`, and the *advanced* file should feel as composed as `single_app_multi_chat_mode.py`.

### No exemptions

Even features that look trivially small (a single LLM provider swap, a one-line system prompt) get an advanced companion. The advanced slot is where you showcase the **parameter variety** of that feature and pair it with **one other pillar customization** so the example earns its place.

