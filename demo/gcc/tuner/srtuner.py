
# define compiler
# define search space
# define benchmark


# Build standalone SRTuner framework
class SRTuner(Tuner):
    def __init__(self, search_space, budget):
        super().__init__(search_space, budget)

