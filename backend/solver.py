import kociemba


def solve_cube(state_string: str) -> str:
    """
    Solve a cube state string in Kociemba order: U, R, F, D, L, B.
    """
    return kociemba.solve(state_string)


def is_solvable_state(state_string: str) -> tuple[bool, str | None, str | None]:
    """
    Validate solvability by attempting to solve.
    Returns (is_solvable, solution_or_none, error_or_none).
    """
    try:
        solution = solve_cube(state_string)
        return True, solution, None
    except Exception as exc:
        return False, None, str(exc)
