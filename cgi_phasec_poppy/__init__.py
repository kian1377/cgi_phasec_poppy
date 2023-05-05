from . import cgi
from importlib import reload
reload(cgi)
from .cgi import CGI
from .proper_cgi import PROPERCGI
from .multi_cgi import multiCGI
from .imshows import imshow1, imshow2, imshow3

__version__ = '0.1.0'

from pathlib import Path
data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
# data_dir = Path('C:/Users/Kian/Documents/data-files/roman-cgi-phasec-data')

