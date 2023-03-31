from . import cgi
from importlib import reload
reload(cgi)
from .cgi import CGI

__version__ = '0.1.0'

