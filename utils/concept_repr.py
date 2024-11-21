import os
import sys
sys.path.append(os.getcwd() + '/..')
from typing import List, Set, Union, Tuple, Callable, Optional, Hashable, Iterable, Any, Literal
from itertools import combinations
from collections import deque
import networkx as nx
from copy import deepcopy
from utils.log_style import Fore, Style
from utils.taxo_utils import Taxonomy
from utils.tokenset_utils import tokenset

class ICONConceptRepresentation:

    def __init__(self):
        pass