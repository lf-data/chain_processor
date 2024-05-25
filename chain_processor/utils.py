import inspect
import random
import string
from typing import Callable, Tuple, List, Dict
import os


class Settings:
    cpus: int = max([int(os.cpu_count()/2), 1]) # Number of CPUs for multiprocessing 
    maxsize: int = 2 # maxsize of lru_cache function

# Function to generate a random string of a given size
def _id_generator(str_size) ->str:
    # Define the characters to use for the string: all ASCII letters
    chars = string.ascii_letters
    # Generate a random id by selecting random characters
    return ''.join(random.choice(chars) for x in range(str_size))

# Function to map input arguments to the node's parameters
def _input_args(args: Tuple, kwargs: Dict, node_args: List) ->Dict:
    # Create a dictionary of keyword arguments that match the node's parameters
    output_args = {node_args[node_args.index(kw)]: kwargs[kw] for kw in kwargs if kw in node_args}
    # If there are no positional arguments, return the keyword arguments
    if len(args) == 0:
        return output_args
    
    # Determine the missing node arguments that need to be filled by positional arguments
    loss_node_arg = [x for x in node_args if x not in output_args]
    if len(loss_node_arg) > 0:
        # Adjust the length of positional arguments if necessary
        if len(args) > len(loss_node_arg):
            args = args[:len(loss_node_arg)]
        elif len(args) < len(loss_node_arg):
            loss_node_arg = loss_node_arg[:len(args)]
        
        # Add the positional arguments to the output dictionary
        output_args |= {y:x for x, y in zip(args, loss_node_arg)}
    return output_args

# Function to check if a function accepts positional or keyword arguments
def _is_positional_or_keyword(func: Callable) ->bool:
    # Get the function's signature
    sig = inspect.signature(func)
    # Iterate through the function's parameters
    for param in sig.parameters.values():
        # Check if the parameter kind is either VAR_POSITIONAL or VAR_KEYWORD
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            return True
    return False

# Function to get the list of argument names of a function
def _get_args(func: Callable) ->bool:
    # Get the function's signature
    sig = inspect.signature(func)
    # Return a list of parameter names
    return [name for name in sig.parameters.keys()]