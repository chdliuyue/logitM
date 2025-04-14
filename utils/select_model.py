from model.PCMNLModel import PCMNLModel


def select_model(config):
    model_type = config['model_type']

    if model_type == 'MNL':
        pass
    elif model_type == 'PCMNL':
        return PCMNLModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

