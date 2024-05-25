import inspect
import random
import string
from typing import Callable, Tuple, List, Dict


def _random_string_generator(str_size) ->str:
    chars = string.ascii_letters
    return ''.join(random.choice(chars) for x in range(str_size))

def _input_args(args: Tuple, kwargs: Dict, node_args: List) ->Dict:
    output_args = {node_args[node_args.index(kw)]: kwargs[kw] for kw in kwargs if kw in node_args}
    if len(args) == 0:
        return output_args
    loss_node_arg = [x for x in node_args if x not in output_args]
    if len(loss_node_arg) > 0:
        if len(args) > len(loss_node_arg):
            args = args[:len(loss_node_arg)]
        elif len(args) < len(loss_node_arg):
            loss_node_arg = loss_node_arg[:len(args)]

        output_args |= {y:x for x, y in zip(args, loss_node_arg)}
    return output_args

def _is_positional_or_keyword(func: Callable) ->bool:
    sig = inspect.signature(func)
    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            return True
    return False

def _get_args(func: Callable) ->bool:
    sig = inspect.signature(func)
    return [name for name in sig.parameters.keys()]