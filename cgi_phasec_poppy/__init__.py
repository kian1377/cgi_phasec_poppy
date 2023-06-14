from . import cgi, cgi_dev, proper_cgi, multi_cgi, math_module

from .cgi import CGI
from.cgi_dev import CGIDEV
from .proper_cgi import PROPERCGI
from .multi_cgi import multiCGI
from .import hlc, hlc_dev

from .math_module import xp, _scipy, ensure_np_array, pad_or_crop
from .imshows import imshow1, imshow2, imshow3

__version__ = '0.1.0'

from pathlib import Path
data_dir = Path('/groups/douglase/kians-data-files/roman-cgi-phasec-data/phasec_data')
# data_dir = Path('C:/Users/Kian/Documents/data-files/roman-cgi-phasec-data')


