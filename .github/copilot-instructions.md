# Copilot Instructions for Chatnificent


## About the package

This is a package for creating LLM chat UIs for developers to customize any way they want.
Developers would be able to add any LLM provider and/or any of their endpoints. The app is ready 
to run out of the box with very minimal setup, so there is no friction involved.


```bash
pip install chatnificent
```

```python
# chatnificent.py
# import dash
# class Chatnificent(dash.Dash):
#     # all method definitions here...

from chatnificent import Chatnificent

app = Chatnificent()

# a default layout is provided but can be customized

if __name__ == "__main__":
    app.run(debug=True)
```

- This is a plotly Dash app that is distributed as a Python package
- Goals:
  - Hackability: The app comes built with several defaults and ways to handle LLM chats,
    responses, dipslaying responses, and many useful methods. Each configuration option
    and/method can be modified/overriden to achieve full customization.
  - Familiarity: Devs should work with the same familiar API of Dash. This class is a
    sub-class of dash.Dash and works exactly the same way. No new learning is required.
  - Full customizability: All layout components and styles, as well as all default callbacks
    can be customized. For example to format messages differently, to customize how to
    display LLM responses that contain images, audio files, etc.
  - Ergonomics: Everything works out of the box, and when you need to make changes, you
    make them one at a time, and you only make the changes that you want.
  - Ease of getting started: As mentioned above. It should takes devs two minutes to get
    the "Hello, world" app running, and then they can work on customizing it.

## Development guidelines

- Any recommendations you provide should be as minimal as possible. Just create the feature you were asked. Nothing more, nothing less.
- Never ever make any UI changes as part of an implementation of a feature, unless asked to do so.
- Your responses are helpful, comprehensive, complete, and don't have unnecessary comments.
- You are smart, critical, helpful, and you challenge what you are asked to do if you think there is a better way of doing things.
- The workflow should be incremental: create the simplest additional concrete feature, master it, making sure all edge cases are handled, create tests, and run several tests to ensure that things are relaiable. Only then you move to a new feature.
- When asked to implement vague or very general tasks, you break them into steps and components, but you also remind me of our policy of incremental development.
- Don't flatter me if you think I made a bad decision, challenge me, provide other options before you rush to coding
- `dash` component IDs should always use snake_case
- `cassName` and CSS IDs should always use kebab-case



## Python-specific guidelines

- We use `uv` for all project and package management tasks
- We run everything using `uv run <module.py>` instead of `python <module.py>`
- Don't worry at all about linting or formatting. This is completely handled by ruff, and is automatically done on save.
- Comments should be included if they explain *why* something is done or clarify a complicated scenario with a simple example. We don't write comments to explain what is being done. The code should be clear.

## Source control and git guidelines

- Never `git add . `
- When adding files to the staging area add them ones that are relevant to the commit, and list them one by one. Don't use `git add .`