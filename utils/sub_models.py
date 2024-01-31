from abc import ABC, abstractmethod
from typing import Any, List, Hashable, Union
from numpy import ndarray
from taxo_utils import Taxonomy

class ICON_ret_model(ABC):
    
    @abstractmethod
    def __call__(self, taxo: Taxonomy, query: str, k: int, *args, **kwargs: Any) -> List[Hashable]:
        pass
    
class ICON_gen_model(ABC):
    
    @abstractmethod
    def __call__(self, labels: List[str], *args, **kwargs: Any) -> str:
        pass
    
class ICON_sub_model(ABC):
    
    @abstractmethod
    def __call__(self, sub: Union[str, List[str]], sup: Union[str, List[str]], *args, **kwargs: Any) -> ndarray:
        pass