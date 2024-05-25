# Chain Processor

Chain Processor is a Python library designed for creating and managing chains of function nodes. This library provides an easy-to-use framework for chaining together functions, layers, and complex processing chains, enabling parallel processing and modular code organization.

## Features

- **Node Decorator**: Convert functions into nodes that can be added to chains.
- **Chain**: Create sequential chains of nodes for step-by-step processing.
- **Layer**: Parallel processing of nodes, either as a list or a dictionary.
- **Automatic Argument Handling**: Automatically manage and pass arguments between nodes.
- **Parallel Processing**: Utilize multiple CPU cores for parallel node execution.
- **Flexible Chain Operations**: Use `>>` and `<<` operators to create complex chains and systems.

## Installation

Install Chain Processor directly from GitHub using pip:

```bash
pip install git+https://github.com/lf-data/chain_processor.git
```

## Usage

### Creating Nodes

Use the `@node` decorator to create nodes from functions.

```python
from chain_processor import node

@node(description="This node adds one to the input number", name="PlusOneNode")
def plus_one(num: int):
    return num + 1

@node(description="This node adds two to the input number", name="PlusTwoNode")
def plus_two(num: int):
    return num + 2

@node(description="This node sums all input numbers", name="SumAllNode")
def sum_all(*args):
    return sum(args)
```

### Creating Chains

Chains are created by using the `>>` and `<<` operators to combine nodes sequentially.

```python
# Basic chain example
base_chain = plus_one >> plus_two >> [plus_one, plus_one] >> sum_all
print(base_chain(1))  # Output: 10

# Intermediate chain example with nodes added before the base chain
intermediate_chain = base_chain << plus_one << plus_two << sum_all
print(intermediate_chain(1))  # Output: 16

# Complex chain example with parallel and sequential operations
chain = plus_one >> [base_chain, plus_two] >> intermediate_chain
print(chain(1))  # Output: 46
```

### Prime Number Chain Example

Create a chain to check if a number is prime and print the result.

```python
@node(description="This node checks if a number is prime", name="PrimeCheckNode")
def give_prime_num(n):
    if n <= 3:
        return n > 1
    if n % 2 == 0 or n % 3 == 0:
        return {"prime": False}
    i = 5
    while i ** 2 <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return {"prime": False}
        i += 6
    return {"prime": True}

@node(description="This node prints if the number is prime", name="PrintPrimeNode")
def print_prime(prime):
    if prime:
        return "This number is prime"
    else:
        return "This number is not prime"

prime_chain = give_prime_num >> print_prime
print(prime_chain(10))  # Output: This number is not prime
```

### Flexible Chain Operations with Named Inputs

Create a chain to greet two names provided as a dictionary.

```python
@node(description="This node greets two names", name="HelloWorldNode")
def hello_world(name1, name2):
    return f"Ciao {name1} e {name2}"

@node(description="This node retrieves the first name", name="GetName1Node")
def get_name1(x1):
    return x1

@node(description="This node retrieves the second name", name="GetName2Node")
def get_name2(x2):
    return x2

chain_name = {"name1": get_name1, "name2": get_name2} >> hello_world
inputs = {"x1": "Marco", "x2": "Francesco"}
print(chain_name(**inputs))  # Output: Ciao Marco e Francesco
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
