import os
import tempfile
from typing import Any

class GlobalSharedScalar:
    def __init__(self, dtype: type, base_path: str, name: str):
        self.dtype = dtype
        self.base_path = base_path
        self.name = name
        self.path = os.path.join(base_path, name)

        os.makedirs(base_path, exist_ok=True)
        if not os.path.exists(self.path):
            self.set(self.dtype(0))

    def get(self):
        with open(self.path, 'r') as f:
            res = f.read().strip()
            if not res:
                # Handle empty file gracefully
                return self.dtype(0)
            return self.dtype(res)

    def set(self, value: Any):
        assert value == 0 or value, f'{value} is not a valid value'
        dir_name = os.path.dirname(self.path)
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False) as tf:
            tf.write(str(value))
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tf.name, self.path)
