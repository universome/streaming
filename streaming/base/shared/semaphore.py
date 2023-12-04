import os
import time
from filelock import FileLock

import numpy as np

from streaming.base.constant import TICK
from streaming.base.shared.scalar import SharedScalar

# Tick to wait in seconds. Should be larger than the lock TICK,
# since otherwise some processes may not be able to acquire the lock to release it.
# (sometimes, there are too few processes waiting for the release and too many waiting for the acquisition)
COUNT_LOCK_ACQUIRE_TICK = TICK * 10


class SharedSemaphore:
    def __init__(self, filelock_root: os.PathLike, name: str, max_access: int):
        self.filelock_path = os.path.join(filelock_root, f'{name}.lock')
        self.count_update_lock = FileLock(self.filelock_path)
        self.max_access = max_access
        with self.count_update_lock:
            self._count = SharedScalar(np.int64, f'{name}_semaphore_count')

    def acquire(self):
        while True:
            with self.count_update_lock:
                count = self._count.get()
                assert count >= 0, f'Semaphore count is negative: {count}'
                if count < self.max_access:
                    self._count.set(count + 1)
                    return
            time.sleep(COUNT_LOCK_ACQUIRE_TICK) # Wait before retrying

    def release(self):
        with self.count_update_lock:
            count = self._count.get()
            assert count >= 1, f'Cannot reduce the semaphore count < 1: {count}'
            self._count.set(count - 1)

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb): # pylint: disable=unused-argument
        self.release()
