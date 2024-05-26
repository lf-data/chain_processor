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

## Examples

### 1. Basic Example with Math Operations

```python
@node(description="Add two numbers")
def add(a, b):
    return a + b

@node(description="Add two numbers")
def divide(a, b):
    return a/b

@node(description="Multiply two numbers")
def multiply(a, b):
    return a * b

chain = [add, divide] >> multiply
print(chain(2, 3))  # Output: 3.333
```

### 2. String Manipulation

```python
from chain_processor import node

@node(description="Convert to uppercase")
def to_upper(text):
    return text.upper()

@node(description="Reverse the string")
def reverse(text):
    return text[::-1]

chain = to_upper >> reverse
print(chain("hello"))  # Output: "OLLEH"
```

### 3. Data Processing with `pandas`

```python
import pandas as pd
from chain_processor import node

@node(description="Create a DataFrame")
def create_df(data):
    return pd.DataFrame(data)

@node(description="Calculate mean of a column")
def calculate_mean(df):
    return df['value'].mean()

@node(description="Calculate the standard deviation of a column")
def calculate_std(df):
    return df['value'].std()

chain = create_df >> [calculate_mean, calculate_std]
data = {'value': [10, 20, 30, 40]}
print(chain(data))  # Output: [25.0, 12.909944487358056]
```

### 4. JSON Processing with `json`

```python
import json
from chain_processor import node

@node(description="Parse JSON")
def parse_json(json_str):
    return {"data": json.loads(json_str)}

@node(description="Extract value by key")
def extract_value(data):
    return data['name']

chain = parse_json  >> extract_value
json_str = '{"name": "Alice", "age": 30}'
print(chain(json_str))  # Output: "Alice"
```

### 5. Web Requests with `requests`

```python
import requests
from chain_processor import node, Node

@node(description="Fetch URL content")
def fetch_url(url):
    return requests.get(url).text

@node(description="Count occurrences of a word")
def count_word(text, word):
    return text.count(word)

chain = {"text":fetch_url, "word": Node(func=lambda word: word)} >> count_word
print(chain(url="https://www.python.org/", word="Python"))  # Output: Number of occurrences of "Python"
```

### 6. Image Processing with `PIL`

```python
@node(description="Open an image")
def open_image(path):
    return Image.open(path)

@node(description="Apply blur filter")
def apply_blur(image):
    return image.filter(ImageFilter.BLUR)

chain = open_image >> apply_blur
image_path = "example.jpg"
blurred_image = chain(image_path)
blurred_image.show()
```

### 7. Data Transformation with `numpy`

```python
import numpy as np
from chain_processor import node

@node(description="Create numpy array")
def create_array(data):
    return np.array(data)

@node(description="Calculate sum of array")
def array_sum(array):
    return np.sum(array)

chain = create_array >> array_sum
data = [1, 2, 3, 4, 5]
print(chain(data))  # Output: 15
```

### 8. Time Manipulation with `datetime`

```python
from datetime import datetime, timedelta
from chain_processor import node

@node(description="Get current time")
def current_time():
    return datetime.now()

@node(description="Add days to datetime")
def add_days(dt):
    return dt + timedelta(days=5)

chain = current_time >> add_days
print(chain())  # Output: Current date + 5 days
```

### 9. Regular Expressions with `re`

```python
import re
from chain_processor import node, Chain

@node(description="Find all matches")
def find_matches(text, pattern):
    return re.findall(pattern, text)

@node(description="Count matches")
def count_matches(matches):
    return len(matches)

chain = find_matches >> count_matches
text = "The rain in Spain stays mainly in the plain."
pattern = r'\bin\b'
print(chain(text, pattern))  # Output: 2
```

### 10. Machine Learning with `scikit-learn`

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from chain_processor import node

@node(description="Load dataset")
def load_data():
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2)

@node(description="Train model")
def train_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

chain = load_data >> train_model
print(chain())  # Output: Model accuracy
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your changes.

## License

This project is licensed under the Apache 2 License. See the LICENSE file for details.
