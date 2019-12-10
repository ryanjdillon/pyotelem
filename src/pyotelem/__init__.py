from . import dives
from . import dsp
from . import dynamics
from . import glides
from . import physio_seal
from . import plots
from . import seawater
from . import utils

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
