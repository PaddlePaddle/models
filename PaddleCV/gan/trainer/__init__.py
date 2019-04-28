import importlib


def get_special_cfg(model_net):
    model = "trainer." + model_net
    modellib = importlib.import_module(model)
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_net.lower():
            model = cls()

    return model.add_special_args
