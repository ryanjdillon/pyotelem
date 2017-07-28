import matplotlib.pyplot as plt
import seaborn
import string

# Use specified style (e.g. 'ggplot')
plt.style.use('seaborn-whitegrid')

# Use specified color palette
colors = seaborn.color_palette()
abc = string.ascii_uppercase

# Global axis properties
linewidth = 0.5

from . import plotdives
from . import plotdsp
from . import plotdynamics
from . import plotglides
from . import plotutils
