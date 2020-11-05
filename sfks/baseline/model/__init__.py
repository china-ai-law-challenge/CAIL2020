from model.qa.qa import Model

model_list = {
    "Model": Model
}


def get_model(model_name):
    """
    Return model instance of model_name.

    Args:
        model_name: (str): write your description
    """
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
