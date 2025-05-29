import dataclasses
from abc import ABC

import control
import numpy as np
import abc

import pandas as pd

_s = control.tf([1, 0], [0, 1])


class DynamicalSystem(abc.ABC):

    @abc.abstractmethod
    def as_state_space(self) -> control.StateSpace:
        raise NotImplementedError

    @abc.abstractmethod
    def simulate_dynamics(self, dt: float, t_final: float) -> pd.DataFrame:
        raise NotImplementedError


@dataclasses.dataclass
class SpringMassDamperSystem(DynamicalSystem, ABC):
    damping_coefficient: float
    natural_frequency: float
    noise_level: float = dataclasses.field(default=1, repr=False)
    state_space: control.StateSpace = dataclasses.field(init=False)

    def __post_init__(self):
        transfer_func = (self.damping_coefficient / (_s ** 2) + (2 * self.damping_coefficient * self.natural_frequency)
                         + self.natural_frequency ** 2)
        state_space_repr = control.tf2ss(transfer_func)
        self.state_space = state_space_repr

    def as_state_space(self) -> control.StateSpace:
        return self.state_space.copy()


    def simulate_dynamics(self, dt: float, t_final: float) -> pd.DataFrame:
        response = self._perform_simulation_raw(dt, t_final)
        result = pd.DataFrame(
            data={
                "t": response.t,
                "u": response.u[0, :],
                "x_1": response.x[0, :],
                "x_2": response.x[0, :],
                "x_1_noisy": response.x[0, :],
                "x_2_noisy": response.x[0, :]
            }, index=list(range(len(response.t)))
        )
        return result

    def _perform_simulation_raw(self, dt, t_final) -> control.TimeResponseData:
        t_sim = np.linspace(0, t_final, round(t_final / dt))
        u = np.zeros_like(t_sim)
        u[t_sim < (len(t_sim) // 8) * dt] = 1
        u[(t_sim < (len(t_sim) // 2) * dt)] = -1
        return control.forced_response(self.state_space, t_sim, U=u)


if __name__ == "__main__":
    simple_msd = SpringMassDamperSystem(
        damping_coefficient=1,
        natural_frequency=10
    )
    print(simple_msd.simulate_dynamics(dt=0.001, t_final=10))
