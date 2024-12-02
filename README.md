# datasette-llm-usage

[![PyPI](https://img.shields.io/pypi/v/datasette-llm-usage.svg)](https://pypi.org/project/datasette-llm-usage/)
[![Changelog](https://img.shields.io/github/v/release/datasette/datasette-llm-usage?include_prereleases&label=changelog)](https://github.com/datasette/datasette-llm-usage/releases)
[![Tests](https://github.com/datasette/datasette-llm-usage/actions/workflows/test.yml/badge.svg)](https://github.com/datasette/datasette-llm-usage/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/datasette/datasette-llm-usage/blob/main/LICENSE)

Track usage of LLM tokens in a SQLite table

This is a **very early alpha**.

## Installation

Install this plugin in the same environment as Datasette.
```bash
datasette install datasette-llm-usage
```
## Usage

This plugin adds functionality to track and manage LLM token usage in Datasette. It creates two tables.

- `_llm_usage`: Tracks usage of LLM tokens
- `_llm_allowance`: Manages credit allowances for LLM usage

### Configuration

By default the tables are created in the internal database passed to Datasette using `--internal internal.db`. You can change that by setting the following in your Datasette plugin configuration:

```json
{
    "plugins": {
        "datasette-llm-usage": {
            "database": "your_database_name"
        }
    }
}
```

### Setting up allowances

Before using LLM models, you need to set up an allowance in the `_llm_allowance` table. You can do this with SQL like:

```sql
insert into _llm_allowance (
    id,
    created,
    credits_remaining,
    daily_reset,
    daily_reset_amount,
    purpose
) values (
    1,
    strftime('%s', 'now')
    10000,
    0,
    0,
    null
);
```
The other columns are not yet used.

### Using the LLM wrapper

The plugin provides an `LLM` class that wraps the `llm` library to track token usage:

```python
from datasette_llm_usage import LLM

llm = LLM(datasette)

# Get available models
models = llm.get_async_models()

# Get a specific model
model = llm.get_async_model("gpt-4o-mini", purpose="my_purpose")

# Use the model
response = await model.prompt("Your prompt here")
text = await response.text()
```
Usage will be automatically recorded.

### Built-in endpoint

The plugin provides a simple demo endpoint at `/-/llm-usage-simple-prompt` that requires authentication and uses the gpt-4o-mini model.

### Supported Models and Pricing

The plugin includes pricing information for various models including:

- Gemini models (1.5-flash, 1.5-pro)
- Claude models (3.5-sonnet, 3-opus, 3-haiku)
- GPT models (gpt-4o, gpt-4o-mini)
- Anthropic models (o1-preview, o1-mini)

Usage is tracked in credits where 1 million credits = 1 cent. Different models have different input and output token costs.

If no database is specified, it will use Datasette's internal database.


## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd datasette-llm-usage
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
