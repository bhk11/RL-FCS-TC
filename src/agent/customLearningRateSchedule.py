from typing import Callable

def linear_schedule(initial_value: float, final_value: float, start_decay: float, stop_decay: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining < start_decay:
            return initial_value
        elif progress_remaining > stop_decay:
            return final_value
        else:
            return ((progress_remaining-start_decay)/(stop_decay-start_decay)) * (initial_value - final_value) + final_value

    return func