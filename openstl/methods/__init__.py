# Copyright (c) CAIRI AI Lab. All rights reserved
from .simvp import SimVP
from .tau import TAU
from .predformer import PredFormer
from .mtdapre import MTDAPre

method_maps = {
    'simvp': SimVP,
    'tau': TAU,
    'predformer': PredFormer,
    'mtdapre': MTDAPre,
}

__all__ = [
    'SimVP', 'TAU', 'PredFormer', 'MTDAPre',
]