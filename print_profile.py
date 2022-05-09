import pstats
from pstats import SortKey


if __name__ == '__main__':
    p = pstats.Stats('profile.txt')
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
    p.strip_dirs().sort_stats(SortKey.TIME).print_stats(10)
