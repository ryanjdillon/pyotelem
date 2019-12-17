from . import dives
from . import dsp
from . import dtag
from . import glides
from . import seawater
from . import utils
from . import plots

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"
