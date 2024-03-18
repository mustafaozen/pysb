import numpy as np

from pysb import *
from pysb.integrate import Solver
from pysb.testing import with_model

@with_model
def test_integrate_with_expression():
    """Ensure a model with Expressions simulates."""

    Monomer('s1')
    Monomer('s9')
    Monomer('s16')
    Monomer('s20')

    # Parameters should be able to contain s(\d+) without error
    Parameter('ks0',2e-5)
    Parameter('ka20', 1e5)

    Initial(s9(), Parameter('s9_0', 10000))

    Observable('s1_obs', s1())
    Observable('s9_obs', s9())
    Observable('s16_obs', s16())
    Observable('s20_obs', s20())

    Expression('keff', (ks0*ka20)/(ka20+s9_obs))

    Rule('R1', None >> s16(), ks0)
    Rule('R2', None >> s20(), ks0)
    Rule('R3', s16() + s20() >> s16() + s1(), keff)

    time = np.linspace(0, 40)

    import ipdb; ipdb.set_trace()

    solver = Solver(model, time, integrator='Rodas4p')
    solver.run()

    assert solver.yexpr_view.shape == (len(time),
                                       len(model.expressions_dynamic()))

    assert solver.yobs_view.shape == (len(time), len(model.observables))


if __name__ == '__main__':
    test_integrate_with_expression()
