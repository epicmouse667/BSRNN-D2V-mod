import wesep.models.convtasnet as convtasnet
import wesep.models.bsrnn as bsrnn
import wesep.models.bsrnn_d2v as bsrnn_d2v


def get_model(model_name: str):
    if model_name.startswith("ConvTasNet"):
        return getattr(convtasnet, model_name)
    elif model_name == "BSRNN":
        return getattr(bsrnn, model_name)
    elif model_name == "BSRNN_D2V":
        return getattr(bsrnn_d2v,model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)