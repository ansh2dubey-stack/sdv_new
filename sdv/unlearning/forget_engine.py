import numpy as np


class ForgetEngine:
    """
    Removes learned influence from the model.
    """

    def remove(self, model, influence_vector):

        # Example simple parameter adjustment
        if hasattr(model, "_model"):
            internal_model = model._model

            if hasattr(internal_model, "weights"):
                internal_model.weights = internal_model.weights - 0.001 * influence_vector

        return model