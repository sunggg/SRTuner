from SRTuner import SRTunerModule
from .common import Tuner

# SRTuner as a standalone tuner
class SRTuner(Tuner):
    def __init__(self, search_space, evaluator):
        super().__init__(search_space, evaluator, "SRTuner")

        # User can customize reward func as Python function and pass to module.
        # In this demo, we use the default reward func. 
        self.mod = SRTunerModule(search_space)

    def generate_candidates(self, batch_size=1):
        return self.mod.generate_candidates()

    def evaluate_candidates(self, candidates):
        return [self.evaluator.evaluate(opt_setting, num_repeats=3) for opt_setting in candidates]

    def reflect_feedback(self, perfs):
        self.mod.reflect_feedback(perfs)
