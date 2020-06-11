from model.qa.qa import Model

model_list = {
    "Model": Model
}


def get_model(model_name):
    if model_name in model_list.keys():
        return model_list[model_name]
    else:
        raise NotImplementedError
