import wesep.models.convtasnet as convtasnet


def get_model(model_name: str):
    if model_name.startswith("ConvTasNet"):
        return getattr(convtasnet, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
