# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Tool Calling — Give the LLM Functions to Execute
=================================================

This example shows how to register Python functions as "tools" that the LLM
can call during a conversation. The LLM decides *when* to call a tool based
on the user's message, executes it, and incorporates the result into its
response.

How It Works
------------
1. Create a ``PythonTool`` instance and register plain Python functions
2. Pass it to ``Chatnificent(tools=...)``
3. The Engine's agentic loop handles the rest:

   a. User sends a message
   b. LLM sees the available tool schemas and may request a tool call
   c. Engine executes the function via the Tools pillar
   d. Tool result is added to the conversation
   e. LLM generates a final answer incorporating the result

Tool Registration
-----------------
``PythonTool`` uses type hints and docstrings to auto-generate the JSON Schema
that the LLM needs. Write your functions with:

- **Type hints** on all parameters (``str``, ``int``, ``float``, ``bool``,
  ``list``, ``dict``)
- A **Numpy-style docstring** describing the function and each parameter

::

    def get_weather(city: str, units: str = "celsius") -> str:
        \"\"\"Get the current weather for a city.

        Parameters
        ----------
        city : str
            The city name to look up.
        units : str
            Temperature units: "celsius" or "fahrenheit".
        \"\"\"
        return f"It's 22°{units[0].upper()} and sunny in {city}."

The schema is generated automatically — no manual JSON needed.

Provider Compatibility
----------------------
Tool calling works with any LLM provider that supports function calling:

- ``chat.llm.OpenAI()``
- ``chat.llm.Anthropic()``
- ``chat.llm.Gemini()`` # bug to be fixed in the upcoming release
- ``chat.llm.DeepSeek()``
- ``chat.llm.OpenRouter()``

Each provider's ``_translate_tool_schema()`` converts the canonical schema
into the provider's native format.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/tool_calling.py

Try asking: "What's the weather in Tokyo?" or "Roll me 3 dice."

What to Explore Next
--------------------
- Register multiple tools — the LLM picks the right one per query
- Combine tools with persistent storage to keep tool-augmented conversations
- Build a custom Tool pillar by subclassing ``chat.tools.Tool``
"""

import random

import chatnificent as chat


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Parameters
    ----------
    city : str
        The city name to look up.
    """
    conditions = random.choice(["sunny", "cloudy", "rainy", "windy"])
    temp = random.randint(15, 35)
    return f"It's {temp}°C and {conditions} in {city}."


def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll one or more dice.

    Parameters
    ----------
    sides : int
        Number of sides on each die.
    count : int
        Number of dice to roll.
    """
    rolls = [random.randint(1, sides) for _ in range(count)]

    return f"Rolled {count}d{sides}: {rolls} (total: {sum(rolls)})"


tools = chat.tools.PythonTool()
tools.register_function(get_weather)
tools.register_function(roll_dice)

app = chat.Chatnificent(
    llm=chat.llm.Anthropic(),
    tools=tools,
)

if __name__ == "__main__":
    app.run()
