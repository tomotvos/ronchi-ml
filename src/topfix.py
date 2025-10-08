from typing import Tuple

def recommend_fix(p_corr: float, tol: float = 0.03) -> Tuple[str, str]:
    """
    Given predicted parabolic correction (1.0 = perfect parabola),
    return a simple next-action suggestion.
    """
    if abs(p_corr - 1.0) <= tol:
        return "no_action", "Parabolic correction near ideal—proceed to finer assessment (zones/edge/astig) if desired."
    if p_corr < 1.0:
        return "increase_correction", "Figure towards deeper parabola (undercorrected)."
    else:
        return "reduce_correction", "Back off correction (overcorrected)."
