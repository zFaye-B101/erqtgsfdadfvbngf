"""Independent numerical checks; SciPy tests are not used by analyse.py."""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from analyse import OUT, lecture_test, read_table, SOURCE95, SOURCE90, t_critical, t_sf


def main():
    for df in [1, 9, 13, 16, 18, 100]:
        for x in [-8, -2, -0.1, 0, 0.1, 2, 8]:
            assert math.isclose(t_sf(x, df), t.sf(x, df), abs_tol=1e-12)
        for alpha in [0.05, 0.025]:
            assert math.isclose(t_critical(alpha, df), t.isf(alpha, df), rel_tol=1e-9)
    tables = {95: read_table(SOURCE95 / 'outer_fold_metrics.csv'),
              90: read_table(SOURCE90 / 'outer_fold_metrics.csv')}
    rows = list(csv.DictReader((OUT / 'all_comparisons.csv').open()))
    for row in rows:
        table = tables[int(row['target'])]
        def vector(group):
            return np.array([table[(group, row['algorithm'], row['condition'], row['subset'], f)][row['metric']]
                             for f in range(1, 11)])
        base, local = vector('F3'), vector(row['local'])
        a, b = (local, base) if row['metric'] == 'ROC_AUC' else (base, local)
        va, vb = a.var(ddof=0), b.var(ddof=0)
        statistic = (a.mean() - b.mean()) / np.sqrt((va + vb) / 10)
        df = math.ceil((va + vb)**2 * 9 / (va**2 + vb**2))
        assert math.isclose(statistic, float(row['statistic']), abs_tol=1e-12)
        assert df == int(row['df'])
        assert math.isclose(t.sf(statistic, df), float(row['p']), abs_tol=1e-12)
    # Check the slide's rounded-summary arithmetic separately from raw data.
    x1 = (10.1 - 9.9) * 3 / math.sqrt(4.1 + 3.2)
    x2 = (11.6 - 9.9) * 3 / math.sqrt(2 + 3.2)
    assert round(x1, 2) == 0.22 and round(x2, 2) == 2.24
    result = {'checks': 'passed', 'rows_verified_independently': len(rows),
              'distribution_checks': 54,
              'slide29_rounded_statistic': x1, 'slide30_rounded_statistic': x2,
              'note': 'Raw-data examples differ slightly because lecture rounds means and variances before substitution.'}
    (OUT / 'independent-validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
