from .output_tool import basic_output_function, null_output_function

output_function_dic = {
    "Basic": basic_output_function,
    "Null": null_output_function
}


def init_output_function(config, *args, **params):
    """
    Initialize a function.

    Args:
        config: (todo): write your description
        params: (dict): write your description
    """
    name = config.get("output", "output_function")

    if name in output_function_dic:
        return output_function_dic[name]
    else:
        raise NotImplementedError
