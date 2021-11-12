# Define constant
FLOAT_MAX=1e100

class FlagInfo:
    def __init__(self, name, configs):
        self.name = name
        self.configs = configs

class Evaluator:
    def __init__(self, path, num_repeats):
        self.path = path
        self.num_repeats = num_repeats
    
    def build(self):
        assert 0, "Undefined"

    def run(self):
        assert 0, "Undefined"

    def evaluate(self):
        assert 0, "Undefined"

    def clean(self):
        assert 0, "Undefined"


class Tuner:
    def __init__(self, search_space, evaluator):
        self.search_space = search_space
        self.evaluator = evaluator
    
    def generate_candidates(self, batch_size=1):
        assert 0, "Undefined"
    
    def evaluate_candidates(self, candidates):
        assert 0, "Undefined"

    def reflect_feedback(perfs):
        assert 0, "Undefined"

    def tune(self, budget, batch_size=1):
        best_opt_setting, best_perf = None, FLOAT_MAX
        i = 0
        while i<budget:
            candidates = self.generate_candidates(batch_size=batch_size)
            perfs = self.evaluate_candidates(candidates)
            self.reflect_feedback(perfs)

            i += len(candidates)
            for opt_setting, perf in zip(candidates, perfs):
                if perf < best_perf:
                    best_perf = perf
                    best_opt_setting = opt_setting
            
            if best_perf == FLOAT_MAX:
                print(f"[{i}] FLOAT MAX")
            else:
                print(f"[{i}] {best_perf:.3f}")
        return best_opt_setting, best_perf