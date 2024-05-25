from .utils import _input_args, _is_positional_or_keyword, _get_args, _random_string_generator
from typing import Any, Optional, Callable, Union, List, Dict, Tuple
from functools import lru_cache
from copy import deepcopy
import json
from multiprocessing.pool import ThreadPool
import os
import inspect

lru_cache()
def _check_input_node(inputs) ->None:
    if isinstance(inputs, (list, tuple, dict)):
        for inp in inputs:
            if isinstance(inputs, dict):
                _check_input_node(inputs[inp])
            else:
                _check_input_node(inp)
    else:
        if not isinstance(inputs, (Chain, Node, Layer)):
            raise TypeError('Si posssono dare in input solo "Node", "Layer", "Chain" o liste delle classi già citate')

lru_cache()
def _convert_parallel_node(inputs) ->Any:
    if isinstance(inputs, (Chain, Node, Layer)):
        return inputs
    else:
        if isinstance(inputs, dict):
            for key in inputs:
                if isinstance(inputs[key], (list, tuple, dict)):
                    inputs[key] = _convert_parallel_node(inputs[key])
        else:
            for i in range(len(inputs)):
                if isinstance(inputs[i], (list, tuple, dict)):
                    inputs[i] = _convert_parallel_node(inputs[i])
        return Layer(inputs)


def node(description: Optional[str] = None, name: Optional[str] = None):
    def run_node(func: Callable):
        return Node(func, description=description, name=name)
    
    return run_node

class BaseChain:
    def add_node(self, *args, **kwargs) ->object:
        return self
    
    def __rshift__(self, other) ->object:
        return self.add_node(other, before=False)
    
    def __rlshift__(self, other) ->object:
        return self.add_node(other, before=False)
    
    def __lshift__(self, other) ->object:
        return self.add_node(other, before=True)
    
    def __rrshift__(self, other) ->object:
        return self.add_node(other, before=True)

class Chain(BaseChain):
    def __init__(self, nodes: List[object]):
        assert len(nodes) > 1, "Ci devono essere almeno due nodi"
        _check_input_node(nodes)
        self._nodes = nodes

    def add_node(self, other, before: bool) ->BaseChain:
        cls = deepcopy(self)
        _check_input_node(other)
        other = _convert_parallel_node(other)
        other = deepcopy(other)
        if before:
            cls._nodes.insert(0, other)
        else:
            cls._nodes.append(other)
        return cls
    
    def __repr__(self) -> str:
        return f"Chain({self._nodes})"
    
    @lru_cache()
    def __call__(self, *args, **kwargs):
        x = None
        for i, node in enumerate(self._nodes):
            if i == 0:
                x = node(*args, **kwargs)
            else:
                if isinstance(x, (list, tuple)):
                    x = node(*x)
                elif isinstance(x, dict):
                    x = node(**x)
                else:
                    x = node(x)
        return x
    

class Layer(BaseChain):
    def __init__(self, nodes: Union[List, Tuple, Dict]):
        assert len(nodes) > 1, "Ci devono essere almeno due nodi"
        assert len([node for node in nodes if isinstance(node, Layer)]) == 0, "Non ci possono essere Layer dentro Layer"
        _check_input_node(nodes)
        self._nodes = nodes
        self._is_dict = True if isinstance(nodes, dict) else False

    def add_node(self, other, before: bool) ->BaseChain:
        cls = deepcopy(self)
        _check_input_node(other)
        other = _convert_parallel_node(other)
        other = deepcopy(other)
        if before:
            chain = Chain(nodes=[other, cls])
        else:
            chain = Chain(nodes=[cls, other])
        return chain
    
    def __call__(self, *args, **kwargs)->Any:
        res = {} if self._is_dict else []
        cpus = 1 if int(os.cpu_count()/2) < 1 else int(os.cpu_count()/2)
        run_node = lambda node, args, kwargs: node(*args, **kwargs)
        with ThreadPool(cpus) as pool:
            if self._is_dict:
                keys = list(self._nodes.keys())
                nodes = list(self._nodes.values())
                input_map = [(node, args, kwargs) for node in nodes]
                output = pool.starmap(run_node, input_map)
                res = {y: x for y, x in zip(keys, output)}
            else:
                input_map = [(node, args, kwargs) for node in self._nodes]
                res = pool.starmap(run_node, input_map)
        return res
    
    def __repr__(self) -> str:
        return f"Layer({self._nodes})"


class Node(BaseChain):
    def __init__(self, 
                 func: Callable,
                 description: Optional[str] = None,
                 name: Optional[str] = None):
        self.positional_or_keyword = _is_positional_or_keyword(func)
        self.name = func.__name__
        self.description = inspect.getdoc(func)
        self.args = _get_args(func)
        if description is not None:
            self.description = description
        if name is not None:
            self.name = name
        self.func = func
        self.id = _random_string_generator(30)
    
    def add_node(self, other, before: bool) ->BaseChain:
        cls = deepcopy(self)
        _check_input_node(other)
        other = _convert_parallel_node(other)
        other = deepcopy(other)
        if before:
            chain = Chain(nodes=[other, cls])
        else:
            chain = Chain(nodes=[cls, other])
        return chain
    
    def __call__(self, *args, **kwargs)-> Any:
        if not self.positional_or_keyword:
            inp_args = _input_args(args, kwargs, node_args=self.args)
            return self.func(**inp_args)
        else:
            return self.func(*args, **kwargs)
    
    def __repr__(self) ->str:
        json_func = json.dumps({'name':self.name, 
                                "description":self.description,
                                "args": self.args})
        return f"Node({json_func})"