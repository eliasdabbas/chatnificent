# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///
"""
Multi-Tool Agent — LLM Picks the Right Tool Automatically
==========================================================

Building on ``tool_calling.py``, this example registers multiple tools and lets
the LLM decide which one(s) to call — and whether to call any at all. The
Engine's agentic loop handles multi-step reasoning:

1. Examine the user's request
2. Call one or more tools (sequentially, across loop iterations)
3. Incorporate results and respond

The Agentic Loop
-----------------
The Engine's ``max_agentic_turns`` parameter (default: 5) controls how many
tool-calling iterations the LLM gets. Each turn:

a. LLM generates a response (may include tool calls)
b. Engine executes the requested tools
c. Tool results are added to the conversation
d. Loop continues until the LLM responds with text (no more tool calls)

If the LLM hits the turn limit, the Engine replies with a "max turns reached"
message.

Designing Good Tools
--------------------
- **Descriptive names**: ``calculate_bmi`` not ``func1``
- **Clear docstrings**: the LLM reads them to decide when to use a tool
- **Type hints on all params**: used to generate the JSON Schema
- **Focused scope**: one tool = one job. Let the LLM compose them.
- **Return strings**: tool results are inserted into the conversation as text

Provider Compatibility
-----------------------
The multi-tool loop works with any LLM that supports function calling. Some
providers handle this better than others — GPT-4o and Claude are excellent at
multi-step tool reasoning.

Running
-------
::

    export OPENAI_API_KEY="sk-..."
    uv run examples/multi_tool_agent.py

Try: "What's the weather in Paris and also roll three dice" — the LLM should
call both tools and combine the results.

What to Explore Next
--------------------
- Adjust ``max_agentic_turns`` on the Engine to allow deeper reasoning chains
- Add error handling in tools (the result string can report errors naturally)
- Combine with ``system_prompt.py`` for a persona that uses tools
- Build a custom Tool pillar for external API integrations
"""

import random
from datetime import datetime, timezone

import chatnificent as chat


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Parameters
    ----------
    city : str
        The name of the city to look up weather for.
    """
    weathers = ["sunny", "cloudy", "rainy", "snowing", "windy"]
    temp = random.randint(-5, 35)
    condition = random.choice(weathers)
    return f"Weather in {city}: {temp}°C, {condition}"


def roll_dice(num_dice: int = 1, sides: int = 6) -> str:
    """Roll one or more dice.

    Parameters
    ----------
    num_dice : int
        How many dice to roll.
    sides : int
        Number of sides per die.
    """
    rolls = [random.randint(1, sides) for _ in range(num_dice)]
    return f"Rolled {num_dice}d{sides}: {rolls} (total: {sum(rolls)})"


def get_current_time(timezone_name: str = "UTC") -> str:
    """Get the current date and time.

    Parameters
    ----------
    timezone_name : str
        The timezone name (only UTC is supported in this demo).
    """
    now = datetime.now(timezone.utc)
    return f"Current time ({timezone_name}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate Body Mass Index.

    Parameters
    ----------
    weight_kg : float
        Weight in kilograms.
    height_m : float
        Height in meters.
    """
    bmi = weight_kg / (height_m**2)
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal weight"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"
    return f"BMI: {bmi:.1f} ({category})"


tools = chat.tools.PythonTool()
tools.register_function(get_weather)
tools.register_function(roll_dice)
tools.register_function(get_current_time)
tools.register_function(calculate_bmi)

app = chat.Chatnificent(
    llm=chat.llm.OpenAI(),
    tools=tools,
)

if __name__ == "__main__":
    app.run()
