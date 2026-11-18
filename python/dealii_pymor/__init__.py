"""pyMOR wrappers for deal.II vectors, operators and models.

Scripts run SPMD, ``mpirun -n N python script.py``: every rank runs the whole
pyMOR script, and the collective operations of deal.II keep the ranks in step.

pyMOR's own MPI support is the opposite model -- rank 0 runs the script and
all other ranks wait in an event loop that pyMOR starts when it is imported.
This package switches that off, so it has to be imported before pyMOR.
"""

import os
import sys

if 'pymor' in sys.modules:
    from pymor.tools import mpi

    if mpi.parallel:
        raise ImportError("dealii_pymor has to be imported before pymor, otherwise pyMOR's MPI "
                          'event loop has already taken over all ranks but rank 0.')
else:
    # Only pymor.tools.mpi reads this: pyMOR then treats every rank as a serial run.
    os.environ['PYMOR_CONFIG_DISABLE'] = ' '.join(
        filter(None, [os.environ.get('PYMOR_CONFIG_DISABLE'), 'MPI']))

    # dealii_core_module finalizes MPI once Python has destroyed the deal.II objects; pyMOR would
    # otherwise have mpi4py finalize it before that.
    try:
        import mpi4py
    except ImportError:
        pass
    else:
        mpi4py.rc.finalize = False

import dealii_core_module  # noqa: E402, F401  initializes MPI

from dealii_pymor.models import DealIIVisualizer, stationary_model  # noqa: E402
from dealii_pymor.operators import DealIIOperator, DealIIRhsOperator, DealIISolver  # noqa: E402
from dealii_pymor.vectorarray import DealIIVector, DealIIVectorSpace  # noqa: E402

__all__ = ['DealIIOperator', 'DealIIRhsOperator', 'DealIISolver', 'DealIIVector', 'DealIIVectorSpace',
           'DealIIVisualizer', 'stationary_model']
