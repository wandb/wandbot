"""This module contains utility functions for the Wandbot application.

This module provides two main functions: `deduplicate` and `format_response`. 
The `deduplicate` function is used to remove duplicates from a list while preserving the order. 
The `format_response` function is used to format the response from the API query for the application

Typical usage example:

  from .utils import deduplicate, format_response

  unique_list = deduplicate(input_list)
  formatted_response = format_response(config, response, outro_message, lang, is_last)
"""

from collections import OrderedDict
from typing import Any, List

from pydantic_settings import BaseSettings
from wandbot.api.schemas import APIQueryResponse


def deduplicate(input_list: List[Any]) -> List[Any]:
    """Remove duplicates from a list while preserving order.

    Args:
        input_list: The list to remove duplicates from.

    Returns:
        A new list with duplicates removed while preserving the original order.
    """
    return list(OrderedDict.fromkeys(input_list))


def format_response(
    config: BaseSettings,
    response: APIQueryResponse | None,
    outro_message: str = "",
    is_last: bool = True,
) -> str:
    """Formats the response from the API query.

    Args:
        :param config: The config object for the app.
        response: The response from the API query.
        outro_message: The outro message to append to the formatted response.
        is_last: Whether the response is the last in a series.

    Returns:
        The formatted response as a string.

    """
    if response is not None:
        result = response.answer
        if "gpt-4" not in response.model:
            warning_message = config.WARNING_MESSAGE.format(
                model=response.model
            )
            result = warning_message + response.answer

        if config.include_sources and response.sources and is_last:
            sources_list = deduplicate(
                [
                    item
                    for item in response.sources.split(",")
                    if item.strip().startswith("http")
                ]
            )
            if len(sources_list) > 0:
                items = min(len(sources_list), 3)
                if config.lang_code == "ja":
                    result = (
                        f"{result}\n\n*参考文献*\n\n>"
                        + "\n> ".join(sources_list[:items])
                        + "\n\n"
                    )
                else:
                    result = (
                        f"{result}\n\n*References*\n\n>"
                        + "\n> ".join(sources_list[:items])
                        + "\n\n"
                    )
        if outro_message:
            result = f"{result}\n\n{outro_message}"

    else:
        result = config.ERROR_MESSAGE
    return result
