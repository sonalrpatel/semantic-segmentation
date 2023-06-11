import os

from train import train
from builders.model_builder import models

if __name__ == "__main__":
    # run train over multiple models
    for model in models.keys():
        train('model', model)