'''
    Implementation of augmentations
    Instructions on adding a new augmentation type

    1. Create a new model file in augmentation_impl and extend
       base.ModelInterface. This involves implementing __init__ and
       transform_audio.

    2. Make sure any data pointed to by the new model are accessible.

    3. Add an entry in the _get_model function below.

    4. Start adding the new block in the json config under the
       augmentation_pipeline block.
'''

from __future__ import absolute_import

from libspeech.augmentation_impl import base
from libspeech.augmentation_impl.walla_noise import WallaNoiseModel
from libspeech.augmentation_impl.impulse_response import ImpulseResponseModel
from libspeech.augmentation_impl.online_bayesian_normalization import \
    OnlineBayesianNormalizationModel
from libspeech.augmentation_impl.resampler import ResamplerModel
from libspeech.augmentation_impl.volume_change import VolumeChangeModel
from libspeech.augmentation_impl.speed_perturb import SpeedPerturbationModel


def _get_model(model_type, params):
    """
    Model factory like the one in Caffe that returns a model instance
    from the given type and parameters.

    Args:
        :param model_type: The augmentation model type
        :type model_type: basestring
        :param params: Parameters used to specify this model
        :type params: dict

    Returns:
        Model of the given type built from the given parameters and name
    """
    print model_type
    print params
    if model_type == "online_bayesian_normalization":
        return OnlineBayesianNormalizationModel(**params)
    elif model_type == "resampler":
        return ResamplerModel(**params)
    elif model_type == "volume_change":
        return VolumeChangeModel(**params)
    elif model_type == "speed_perturb":
        return SpeedPerturbationModel(**params)
    elif model_type == "walla_noise":
        return WallaNoiseModel(**params)
    elif model_type == "impulse_response":
        return ImpulseResponseModel(**params)
    else:
        raise ValueError("Unknown augmentation model type")


def parse_pipeline_from(config_blocks):
    """
    Parses the configuration file and builds a chain of models

    Args:
        :parma config_blocks: Loaded json configuration
        :type config_blocks: list of dict

    Returns:
        augmentation pipeline (base.AugmentationPipeline); use the
        transform_audio method to add effects.
    """
    models = []
    rates = []
    if len(config_blocks) == 0:
        models = [base.IdentityModel()]
        rates = [lambda iteration: 1.0]
    else:
        for block in config_blocks:
            params = {
                key: base.parse_parameter_from(val)
                for key, val in block.items() if key != "type" and key != "rate"
            }
            model = _get_model(block["type"], params)
            models.append(model)
            rates.append(base.parse_parameter_from(block["rate"]))
    return base.AugmentationPipeline(models, rates)
