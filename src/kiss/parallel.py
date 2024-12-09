"""
This module contains the classes and functions used for parallelism.
"""

from multiprocessing.managers import SharedMemoryManager, SyncManager
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from threading import Lock
from typing import Union, Type


class Queue:
    """
    Wrapper around the threading Queue class to prevent race conditions.
    """

    def __init__(self, queue: SyncManager, lock: SyncManager) -> None:
        """
        Parameters
        ----------
        queue: SyncManager
            SyncManager instance responsible for Queue instances.
        lock: SyncManager
            SyncManager instance responsible for Lock instances.
        """
        self._queue = queue.Queue()
        self._lock = lock.Lock()

    def put(self, *args, **kwargs) -> None:
        """
        Put the given function args and kwargs in the queue as a tuple.
        """
        with self._lock:
            self._queue.put(item=(args, kwargs))

    def get(self) -> tuple:
        """
        Get a tuple of function args and kwargs from the queue.

        Returns
        -------
        item: tuple
            Function args and kwargs.

        Raises
        ------
        ValueError
            Queue is empty.
        """
        with self._lock:
            try:
                items = self._queue.get(block=False)
            except Empty:
                items = None

        if items is None:
            raise ValueError("Queue is empty.")

        return items


class Memory:
    SUPPORTED = Union[str, int, float]

    def __init__(self, memory: SharedMemoryManager, lock: SyncManager, type_: Type[SUPPORTED]) -> None:
        _sizes = {str: 128, int: 4, float: 8}
        _reads = {str: self._read_str, int: self._read_int}
        _writes = {str: self._write_str, int: self._write_int}

        self._size = _sizes[type_]
        self._read = _reads[type_]
        self._write = _writes[type_]

        self._memory = memory.SharedMemory(size=self._size)
        self._lock = lock.Lock()

    def _read_str(self, encoding: str = "utf-8") -> str:
        with self._lock:
            memory = SharedMemory(name=self._memory.name, create=False)
            item = memory.buf.tobytes().decode(encoding).rstrip("\x00")
            memory.close()

        return item

    def _write_str(self, item: str, encoding: str = "utf-8") -> None:
        i_bytes = item.encode(encoding)
        i_length = len(i_bytes)

        with self._lock:
            shared_error = SharedMemory(name=self._memory.name, create=False)
            shared_error.buf[:i_length] = i_bytes[:i_length]
            shared_error.close()

    def _read_int(self) -> int:
        with self._lock:
            memory = SharedMemory(name=self._memory.name, create=False)
            item = memory.buf[0]
            memory.close()

        return item

    def _write_int(self, item: int) -> None:
        with self._lock:
            memory = SharedMemory(name=self._memory.name, create=False)
            memory.buf[0] = item
            memory.close()

    def read(self):
        return self._read()

    def write(self, item):
        self._write(item)


class ManagerShared:
    """
    Wrapper around the multiprocessing manager classes to facilitate management of shared memory, locks, queues,
    and processes.
    """

    def __init__(self) -> None:
        self._manager_memory = SharedMemoryManager()
        self._manager_memory.start()

        self._manager_locks = SyncManager()
        self._manager_locks.start()

        self._manager_queues = SyncManager()
        self._manager_queues.start()

    def __del__(self) -> None:
        self._manager_memory.shutdown()
        self._manager_locks.shutdown()
        self._manager_queues.shutdown()

    def request_lock(self) -> Lock:
        """
        Request a lock which can be shared across processes.

        Returns
        -------
        lock: Lock
            SyncManager Lock instance which can be shared across processes.
        """
        return self._manager_locks.Lock()

    def request_queue(self) -> Queue:
        """
        Request a queue which can be shared across processes.

        Returns
        -------
        queue: Queue
            Queue instance which can be shared across processes.
        """
        return Queue(queue=self._manager_queues, lock=self._manager_locks)

    def request_memory(self, type_: Type[Memory.SUPPORTED]):
        return Memory(memory=self._manager_memory, lock=self._manager_locks, type_=type_)


if __name__ == "__main__":
    manager = ManagerShared()
    memory_str = manager.request_memory(type_=str)
    memory_str.write(item="Hello!")
    print(memory_str.read())

    memory_int = manager.request_memory(type_=int)
    memory_int.write(item=10)
    print(memory_int.read())
