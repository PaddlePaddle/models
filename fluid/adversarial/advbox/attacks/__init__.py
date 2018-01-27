"""
Attack methods
"""

from .base import Attack
from .deepfool import DeepFoolAttack
from .gradientsign import FGSM
from .gradientsign import GradientSignAttack
from .iterator_gradientsign import IFGSM
from .iterator_gradientsign import IteratorGradientSignAttack
