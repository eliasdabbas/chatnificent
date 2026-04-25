# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "chatnificent[openai]",
# ]
# ///

"""
How to Call Functions with Chat Models — From Cookbook to Production
====================================================================

This is a production implementation of the OpenAI Cookbook example:
https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb

The cookbook is an excellent teaching resource. It walks through function calling
step by step: defining JSON schemas by hand, managing the message loop manually,
and using ``tenacity`` for retries and ``termcolor`` for coloured CLI output. It's
the right way to learn what's happening under the hood.

This file is its production counterpart. The conversation loop, tool dispatch,
streaming, and message persistence are all handled by Chatnificent. The only code
here is the code that's actually yours: the weather tools and the system prompt.

What the cookbook needs that this file doesn't
----------------------------------------------
- ``tenacity`` — retry decorator for transient API errors
- ``termcolor`` — coloured CLI output to make tool calls visible
- ``pretty_print_conversation()`` — ~20-line helper to render message history
- Hand-written JSON schemas — one per function, ~15 lines each
- A manual ``messages`` list and ``messages.append(...)`` loop
- Explicit tool dispatch: inspect ``finish_reason``, call the function, append
  the result, call the API again

What this file adds that the cookbook doesn't
---------------------------------------------
- A real weather API (Open-Meteo) — no mocked random responses
- A persistent chat UI served at http://127.0.0.1:7777
- Streaming responses out of the box
- Conversation history that survives across turns

How it works
------------
``PythonTool`` reads the type hints and Numpy-style docstrings on
``get_current_weather`` and ``get_n_day_weather_forecast`` and generates the JSON
schemas automatically. ``WeatherAI`` subclasses ``chat.llm.OpenAI`` to prepend the
system prompt on every request without touching the conversation store.

The weather data comes from the Open-Meteo forecast API (https://open-meteo.com),
a free, no-key-required service. ``_geocode`` resolves any city or neighbourhood
name to coordinates using the Open-Meteo geocoding API, then the forecast API
returns real current conditions or a multi-day outlook.

Run
---
    uv run openai_cookbook/How_to_call_functions_with_chat_models.py

Then open http://127.0.0.1:7777 and try:

- "What's the weather in Tokyo?"
- "Give me a 5-day forecast for London."
- "Is it warmer in Lisbon or Madrid right now?"
"""

import json
import urllib.parse
import urllib.request

import chatnificent as chat

SYSTEM_PROMPT = """Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
For locations, a city or neighbourhood name is always sufficient — never ask for a zip code or more specific address.

When presenting weather forecasts, format the results as a markdown table with columns: Date, Low, High, and Condition.
When presenting current weather, provide it in natural language, use the Condition value with any accompanying emojis if available.
Always copy the Condition value exactly as returned by the tool — do not rephrase or remove any characters."""

WMO_DESCRIPTIONS = {
    0: "☀️ Clear Sky",
    1: "🌤️ Mainly Clear",
    2: "⛅ Partly Cloudy",
    3: "☁️ Overcast",
    45: "🌫️ Fog",
    48: "🌫️ Icy Fog",
    51: "🌦️ Light Drizzle",
    53: "🌦️ Moderate Drizzle",
    55: "🌧️ Dense Drizzle",
    61: "🌧️ Light Rain",
    63: "🌧️ Moderate Rain",
    65: "🌧️ Heavy Rain",
    71: "🌨️ Light Snow",
    73: "🌨️ Moderate Snow",
    75: "❄️ Heavy Snow",
    77: "🌨️ Snow Grains",
    80: "🌦️ Light Showers",
    81: "🌦️ Moderate Showers",
    82: "⛈️ Violent Showers",
    85: "🌨️ Light Snow Showers",
    86: "🌨️ Heavy Snow Showers",
    95: "⛈️ Thunderstorm",
    96: "⛈️ Thunderstorm With Hail",
    99: "⛈️ Thunderstorm With Heavy Hail",
}


def _geocode(location: str) -> tuple:
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": location, "count": 1}
    )
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())
    results = data.get("results", [])
    if not results:
        raise ValueError(f"Location not found: {location}")
    return results[0]["latitude"], results[0]["longitude"]


def get_current_weather(location: str, format: str = "celsius") -> str:
    """Get the current weather for a location.

    Parameters
    ----------
    location : str
        The city or neighbourhood name, e.g. 'San Francisco' or 'Upper East Side'.
    format : str
        The temperature unit to use. Infer this from the location. One of:
        'celsius', 'fahrenheit'.
    """
    lat, lon = _geocode(location)
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weathercode,windspeed_10m",
            "temperature_unit": format,
            "timezone": "auto",
        }
    )
    with urllib.request.urlopen(
        "https://api.open-meteo.com/v1/forecast?" + params
    ) as resp:
        data = json.loads(resp.read())
    current = data["current"]
    temp = current["temperature_2m"]
    unit = "°F" if format == "fahrenheit" else "°C"
    desc = WMO_DESCRIPTIONS.get(current["weathercode"], "unknown")
    wind = current["windspeed_10m"]
    return f"{location}: {temp}{unit}, {desc}, wind {wind} km/h."


def get_n_day_weather_forecast(
    location: str, num_days: int, format: str = "celsius"
) -> str:
    """Get an N-day weather forecast for a location.

    Parameters
    ----------
    location : str
        The city or neighbourhood name, e.g. 'San Francisco' or 'Upper East Side'.
    num_days : int
        The number of days to forecast.
    format : str
        The temperature unit to use. Infer this from the location. One of:
        'celsius', 'fahrenheit'.
    """
    lat, lon = _geocode(location)
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,weathercode",
            "temperature_unit": format,
            "timezone": "auto",
            "forecast_days": num_days,
        }
    )
    with urllib.request.urlopen(
        "https://api.open-meteo.com/v1/forecast?" + params
    ) as resp:
        data = json.loads(resp.read())
    daily = data["daily"]
    unit = "°F" if format == "fahrenheit" else "°C"
    lines = [
        f"{daily['time'][i]}: {daily['temperature_2m_min'][i]}{unit}–{daily['temperature_2m_max'][i]}{unit}, {WMO_DESCRIPTIONS.get(daily['weathercode'][i], 'unknown')}"
        for i in range(num_days)
    ]
    return f"{num_days}-day forecast for {location}:\n" + "\n".join(lines)


class WeatherAI(chat.llm.OpenAI):
    def generate_response(self, messages, **kwargs):
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        return super().generate_response(messages, **kwargs)


tools = chat.tools.PythonTool()
tools.register_function(get_current_weather)
tools.register_function(get_n_day_weather_forecast)

app = chat.Chatnificent(
    llm=WeatherAI(),
    tools=tools,
)

if __name__ == "__main__":
    app.run()
