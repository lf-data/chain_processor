from .utils import _input_args, _is_positional_or_keyword, _get_args, _id_generator, Settings
from typing import Any, Optional, Callable, Union, List, Dict, Tuple
from functools import lru_cache
from copy import deepcopy
import json
from multiprocessing.pool import ThreadPool
import inspect

lru_cache(maxsize=Settings.maxsize)
def _check_input_node(inputs) ->None:
    # If inputs is a list, tuple, or dict, iterate through each element
    if isinstance(inputs, (list, tuple, dict)):
        for inp in inputs:
            # If inputs is a dict, check each value
            if isinstance(inputs, dict):
                _check_input_node(inputs[inp])
            else:
                # Otherwise, check each item in the list or tuple
                _check_input_node(inp)
    else:
        # If input is not a Chain, Node, or Layer, raise a TypeError
        if not isinstance(inputs, (Chain, Node, Layer)):
            raise TypeError('Only "Node", "Layer", "Chain", or lists of these classes can be used as inputs')

lru_cache(maxsize=Settings.maxsize)
def _convert_parallel_node(inputs) ->Any:
    # If inputs is a Chain, Node, or Layer, return it as is
    if isinstance(inputs, (Chain, Node, Layer)):
        return inputs
    else:
        if isinstance(inputs, dict):
            # If inputs is a dict, convert each value
            for key in inputs:
                if isinstance(inputs[key], (list, tuple, dict)):
                    inputs[key] = _convert_parallel_node(inputs[key])
        else:
            # If inputs is a list or tuple, convert each element
            for i in range(len(inputs)):
                if isinstance(inputs[i], (list, tuple, dict)):
                    inputs[i] = _convert_parallel_node(inputs[i])
        return Layer(inputs)

# Decorator function to create a node
def node(description: Optional[str] = None, name: Optional[str] = None):
    def run_node(func: Callable):
        return Node(func, description=description, name=name)
    
    return run_node

class BaseChain:
    # Method to add a node to the chain

    def add_node(self, *args, **kwargs) ->"BaseChain":
        return self
    
    # Overloading the >> operator to add a node after the current chain
    def __rshift__(self, other) ->"BaseChain":
        return self.add_node(other, before=False)
    
    # Overloading the << operator to add a node before the current chain
    def __rlshift__(self, other) ->"BaseChain":
        return self.add_node(other, before=False)
    
    # Overloading the << operator to add a node before the current chain
    def __lshift__(self, other) ->"BaseChain":
        return self.add_node(other, before=True)
    
    # Overloading the >> operator to add a node after the current chain
    def __rrshift__(self, other) ->"BaseChain":
        return self.add_node(other, before=True)

class Chain(BaseChain):
    def __init__(self, nodes: List[object]):
        # Ensure there are at least two nodes in the chain
        assert len(nodes) > 1, "There must be at least two nodes"
        _check_input_node(nodes)
        self._nodes = nodes

    def add_node(self, other, before: bool) ->BaseChain:
        # Create a deep copy of the current instance to avoid modifying the original
        cls = deepcopy(self)
        # Check if the input node is valid
        _check_input_node(other)
        # Convert the input node into layer if one is a list, tuple or dict
        other = _convert_parallel_node(other)
        # Create a deep copy of the input node to avoid modifying the original
        other = deepcopy(other)
        # Insert the node at the beginning if 'before' is True, otherwise append it at the end
        if before:
            cls._nodes.insert(0, other)
        else:
            cls._nodes.append(other)
        return cls
    
    def __call__(self, *args, **kwargs):
        # Initialize a variable to store the output of the nodes
        x = None
        # Iterate over the nodes in the chain
        for i, node in enumerate(self._nodes):
            # For the first node, pass the arguments directly
            if i == 0:
                x = node(*args, **kwargs)
            else:
                # For subsequent nodes, process the output from the previous node
                if isinstance(x, (list, tuple)):
                    # If the output is a list or tuple, unpack it as positional arguments
                    x = node(*x)
                elif isinstance(x, dict):
                    # If the output is a dictionary, unpack it as keyword arguments
                    x = node(**x)
                else:
                    # Otherwise, pass the output as a single argument
                    x = node(x)
        return x
    
    def __repr__(self) -> str:
        return f"Chain({self._nodes})"
    

class Layer(BaseChain):
    def __init__(self, nodes: Union[List, Tuple, Dict]):
        # Ensure there are at least two nodes in the layer
        assert len(nodes) > 1, "There must be at least two nodes"
        # Ensure that there are no nested layers within this layer
        assert len([node for node in nodes if isinstance(node, Layer)]) == 0, "Layers cannot contain other Layers"
        # Check if the input nodes are valid
        _check_input_node(nodes)
        # Store the nodes in the layer
        self._nodes = nodes
        # Determine if the nodes are stored in a dictionary
        self._is_dict = True if isinstance(nodes, dict) else False

    def add_node(self, other, before: bool) ->BaseChain:
        # Create a deep copy of the current instance to avoid modifying the original
        cls = deepcopy(self)
        # Check if the input node is valid
        _check_input_node(other)
        # Convert the input node into layer if one is a list, tuple or dict
        other = _convert_parallel_node(other)
        # Create a deep copy of the input node to avoid modifying the original
        other = deepcopy(other)
        # Insert the node before or after the current layer based on the 'before' flag and create a Chain
        if before:
            chain = Chain(nodes=[other, cls])
        else:
            chain = Chain(nodes=[cls, other])
        return chain
    
    def __call__(self, *args, **kwargs)->Any:
        # Initialize the result container as a dictionary if nodes are stored in a dictionary, otherwise as a list
        res = {} if self._is_dict else []
        # Determine the number of CPU cores to use, at least 1 and at most half of the available cores
        cpus = Settings.cpus
        # Function to run a node with given arguments, used into Thread Pool
        run_node = lambda node, args, kwargs: node(*args, **kwargs)
        # Use a thread pool to parallelize the execution of nodes
        with ThreadPool(cpus) as pool:
            if self._is_dict:
                # If nodes are stored in a dictionary, create a mapping of nodes to their arguments
                keys = list(self._nodes.keys())
                nodes = list(self._nodes.values())
                input_map = [(node, args, kwargs) for node in nodes]
                # Execute the nodes in parallel and store the results in a dictionary
                output = pool.starmap(run_node, input_map)
                res = {y: x for y, x in zip(keys, output)}
            else:
                # If nodes are stored in a list or tuple, create a mapping of nodes to their arguments
                input_map = [(node, args, kwargs) for node in self._nodes]
                # Execute the nodes in parallel and store the results in a list
                res = pool.starmap(run_node, input_map)
        return res
    
    def __repr__(self) -> str:
        return f"Layer({self._nodes})"


class Node(BaseChain):
    def __init__(self, 
                 func: Callable,
                 description: Optional[str] = None,
                 name: Optional[str] = None):
        # Determine if the function accepts positional or keyword arguments
        self.positional_or_keyword = _is_positional_or_keyword(func)
        # Set the name of the node to the function's name
        self.name = func.__name__
        # Get the function's docstring as its description
        self.description = inspect.getdoc(func)
        # Retrieve the function's argument names
        self.args = _get_args(func)
        # If a custom description is provided, use it
        if description is not None:
            self.description = description
        # If a custom name is provided, use it
        if name is not None:
            self.name = name
        # Store the function to be executed by the node
        self.func = func
        # Generate a random ID for the node
        self.id = _id_generator(30)
    
    def add_node(self, other, before: bool) ->BaseChain:
        # Create a deep copy of the current node to avoid modifying the original
        cls = deepcopy(self)
        # Check if the input node is valid
        _check_input_node(other)
        # Convert the input node into layer if one is a list, tuple or dict
        other = _convert_parallel_node(other)
        # Create a deep copy of the input node to avoid modifying the original
        other = deepcopy(other)
        # Insert the node before or after the current node based on the 'before' flag
        if before:
            chain = Chain(nodes=[other, cls])
        else:
            chain = Chain(nodes=[cls, other])
        return chain
    
    def __call__(self, *args, **kwargs)-> Any:
        # If the function does not accept positional arguments
        if not self.positional_or_keyword:
            # Map the input arguments to the function's parameters
            inp_args = _input_args(args, kwargs, node_args=self.args)
            # Call the function with keyword arguments
            return self.func(**inp_args)
        else:
            # Call the function with positional arguments
            return self.func(*args, **kwargs)
    
    def __repr__(self) ->str:
        json_func = json.dumps({'name':self.name, 
                                "description":self.description,
                                "args": self.args})
        return f"Node({json_func})"