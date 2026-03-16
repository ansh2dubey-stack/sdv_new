import numpy as np


class UnlearningVerifier:
    """
    Verify that a row has been forgotten.
    """

    def verify(self, patient_vector):

        norm = np.linalg.norm(patient_vector)

        if norm < 1e-3:
            return True

        return False