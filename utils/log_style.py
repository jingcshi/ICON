from dataclasses import dataclass

@dataclass
class Style:
    RESET_ALL = '\x1b[0m'
    BRIGHT = '\x1b[1m'
    DIM = '\x1b[2m'
    NORMAL = '\x1b[22m'
    
@dataclass
class Fore:    
    BLACK = '\x1b[30m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    BLUE = '\x1b[34m'
    MAGENTA = '\x1b[35m'
    CYAN = '\x1b[36m'
    WHITE = '\x1b[37m'
    RESET = '\x1b[39m'
    
@dataclass
class Back:    
    BLACK = '\x1b[40m'
    RED = '\x1b[41m'
    GREEN = '\x1b[42m'
    YELLOW = '\x1b[43m'
    BLUE = '\x1b[44m'
    MAGENTA = '\x1b[45m'
    CYAN = '\x1b[46m'
    WHITE = '\x1b[47m'
    RESET = '\x1b[49m'