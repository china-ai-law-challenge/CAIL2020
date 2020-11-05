import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    """
    Null function to null value

    Args:
        data: (array): write your description
        config: (todo): write your description
        params: (dict): write your description
    """
    return ""


def basic_output_function(data, config, *args, **params):
    """
    Basic function to write basic function.

    Args:
        data: (array): write your description
        config: (todo): write your description
        params: (dict): write your description
    """
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True)
