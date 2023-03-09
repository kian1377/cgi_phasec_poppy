from . import cgi
from importlib import reload
reload(cgi)
from .cgi import CGI

__version__ = '0.1.0'

from pathlib import Path
# data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data')
data_dir = Path('C:/Users/Kian/Documents/data-files/roman-cgi-phasec-data')

