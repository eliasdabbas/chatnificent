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

## Package Architecture

The following modules comprise the architecture of the package:

src/
└── chatnificent/
    ├── __init__.py
    ├── action_handlers.py
    ├── auth_managers.py
    ├── conversation_managers.py
    ├── knowledge_retrievers.py
    ├── layout_builders.py
    ├── llm_providers.py
    ├── message_formatters.py
    ├── models.py
    ├── persistence_managers.py
    ├── py.typed
    └── themes.py


We are following strict guidlines, especially separation of concerns and following the
UNIX philosophy of having each component do one thing and do it well.

It is important to follow the following rule: every method of every one of these classes
should only be concerned with what it is supposed to do (read_messages, get_current_user, etc.).
When we want composite functionality, it should be orchestrated by callbacks or other functions/methods.
We don't want to pollute the atomic methods with anything other than what they actually do.

### Most Important BaseClasses

`auth_managers.py`

```python
class BaseAuthManager(ABC):
    """Interface for identifying the current user."""

    @abstractmethod
    def get_current_user_id(self, **kwargs) -> str:
        """Determines and returns the ID of the current user."""
        pass
```

`conversation_managers.py`:

```python
class BaseConversationManager(ABC):
    """Interface for managing conversation lifecycle."""

    @abstractmethod
    def list_conversations(self, user_id: str) -> List[Dict[str, str]]:
        """Lists all conversations for a given user."""
        pass

    @abstractmethod
    def get_next_conversation_id(self, user_id: str) -> str:
        """Generates a new, unique conversation ID for a user."""
        pass

```

`layout_builders.py`

```python
class BaseLayoutBuilder(ABC):
    """Interface for building the Dash component layout."""

    @abstractmethod
    def build_layout(self) -> DashComponent:
        """Constructs and returns the entire Dash component tree for the UI."""
        pass

# check this concrete implementation when needed:
class DefaultLayoutBuilder(BaseLayoutBuilder):

```

`llm_providers.py`

```python
class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM providers."""

    @abstractmethod
    def generate_response(
        self, messages: List[Dict[str, Any]], model: str, **kwargs: Any
    ) -> Union[Any, Iterator[Any]]:
        """Generates a response from the LLM provider.

        Parameters
        ----------
        messages : List[Dict[str, Any]]
            A list of message dictionaries, conforming to the provider's
            expected format.
        model : str
            The specific model to use for the generation.
        **kwargs : Any
            Provider-specific parameters (e.g., stream, temperature) to be
            passed directly to the SDK.

        Returns
        -------
        Union[Any, Iterator[Any]]
            The provider's native, rich response object for a non-streaming
            call, or an iterator of native chunk objects for a streaming call.
        """
        pass

```

`message_formatters.py

```python
class BaseMessageFormatter(ABC):
    """Interface for converting message data into Dash components."""

    @abstractmethod
    def format_messages(self, messages: List[ChatMessage]) -> List[DashComponent]:
        """Converts a list of message models into renderable Dash components."""
        pass

```

`models.py`

```python
class ChatMessage(BaseModel):
    """Represents a single message within a conversation."""

    role: Role
    content: str

class Conversation(BaseModel):
    """Represents a complete chat conversation session."""

    messages: List[ChatMessage] = Field(default_factory=list)
```

`persistence_managers.py`

```python
class BasePersistenceManager(ABC):
    """Interface for saving and loading conversation data."""

    @abstractmethod
    def load_conversation(self, convo_id: str, user_id: str) -> Conversation:
        """Loads a single conversation from the persistence layer."""
        pass

    @abstractmethod
    def save_conversation(self, conversation: Conversation, user_id: str):
        """Saves a single conversation to the persistence layer."""
        pass
```

## Python-specific guidelines

- We use `uv` for all project and package management tasks
- We run everything using `uv run <module.py>` instead of `python <module.py>`
- Don't worry at all about linting or formatting. This is completely handled by ruff, and is automatically done on save.
- Comments should be included if they explain *why* something is done or clarify a complicated scenario with a simple example. We don't write comments to explain what is being done. The code should be clear.

## Source control and git guidelines

- Never `git add . `
- When adding files to the staging area add them ones that are relevant to the commit, and list them one by one. Don't use `git add .`