import cProfile
from pstats import Stats
stats = Stats("/scratch/pyllm/dhimoila/output/test_profiling/profile_step1.txt")
stats.strip_dirs().sort_stats("cumulative").print_stats(40)