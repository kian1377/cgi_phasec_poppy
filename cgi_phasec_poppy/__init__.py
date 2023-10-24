from . import cgi, proper_cgi, parallelized_cgi, math_module

from .cgi import CGI
from .proper_cgi import PROPERCGI
from .parallelized_cgi import ParallelizedCGI
from .source_flux import SOURCE

from .math_module import xp, _scipy, ensure_np_array
from .imshows import imshow1, imshow2, imshow3

__version__ = '0.1.0'

from pathlib import Path
# data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
data_dir = Path('/home/kianmilani/Projects/roman-cgi-phasec-data')
# data_dir = Path('C:/Users/Kian/Documents/data-files/roman-cgi-phasec-data')


