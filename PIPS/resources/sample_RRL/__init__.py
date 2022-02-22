import importlib.resources as pkg_resources
from PIPS.utils.connect_LPP import data_readin_LPP

samples = []
for f in pkg_resources.contents(__package__):
    if f in ['__pycache__','__init__.py']:
        continue
    with pkg_resources.path(__package__,f) as p:
        samples.append(data_readin_LPP(p))
