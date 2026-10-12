"""Bounded temperature range expansion using inner validation only."""
import math


def search_temperatures(evaluate, initial, maximum_extensions, factor, bounds):
    """Reuse evaluated candidates; keep three values and prefer the middle on ties."""
    grid = sorted(float(value) for value in initial)
    records, history = {}, []
    for stage in range(maximum_extensions + 1):
        for value in grid:
            if value not in records:
                records[value] = evaluate(value)
                if not math.isfinite(records[value]['mean_auc']):
                    raise ValueError('Temperature validation AUC must be finite.')
        best = max([grid[1], grid[0], grid[2]], key=lambda value: records[value]['mean_auc'])
        history.append({'stage': stage, 'grid': grid.copy(), 'selected_temperature': best,
                        'mean_validation_auc': {str(value): records[value]['mean_auc'] for value in grid}})
        if best == grid[1] or stage == maximum_extensions:
            break
        if best == grid[0]:
            value = max(bounds[0], grid[0] / factor)
            if value >= grid[0]:
                break
            grid = [value, grid[0], grid[1]]
        else:
            value = min(bounds[1], grid[2] * factor)
            if value <= grid[2]:
                break
            grid = [grid[1], grid[2], value]
    return best, grid, records, history
