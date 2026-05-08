"""Templates package — bundled HTML/CSS/JS templates and their contract.

Each subdirectory here is a self-contained template (e.g. ``default/``)
holding ``template.html``, ``styles.css``, ``scripts.js``, and a ``vendor/``
subfolder with pinned third-party libraries. The contract that every shipped
template must honor lives in :mod:`chatnificent.templates._contract`.
"""

from . import _contract

from . import _contract
