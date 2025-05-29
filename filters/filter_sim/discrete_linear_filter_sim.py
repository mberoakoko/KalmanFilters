import numpy as np
import control
from filters.filter_impl.discrete_linear_filters import FilterBase, DiscreteLinearKalmanFilter
from systems.dynamical_systems import DynamicalSystem, SpringMassDamperSystem

def test_linear_kalman_filter() -> None:
    simple_msd = SpringMassDamperSystem(
        damping_coefficient=1,
        natural_frequency=0.5
    )
    sys_as_state_space: control.StateSpace = simple_msd.as_state_space()
    print(sys_as_state_space)
    kalman_filter = DiscreteLinearKalmanFilter(
        system= sys_as_state_space,
        convariance_matrix=np.eye(len(sys_as_state_space.A)),
        forcing_convariance=np.eye(1),
    )

    response = control.step_response(sys_as_state_space, 1)
    y_sample = response.y[0, :, :].T
    u_sample = np.ones(shape=(len(y_sample), 1))
    print(response.y.shape)
    print(y_sample.shape)
    print(u_sample.shape)
    for u, y in zip(u_sample, y_sample):
        print(kalman_filter.evaluate(y, u))

if __name__ == "__main__":
    test_linear_kalman_filter()