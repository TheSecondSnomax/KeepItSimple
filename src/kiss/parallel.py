"""
This module contains the classes and functions used for parallelism.
"""

from multiprocessing.managers import SharedMemoryManager, SyncManager
from multiprocessing.shared_memory import SharedMemory
from queue import Empty
from threading import Lock
from typing import Type, Union


class Queue:
    """
    Wrapper around the threading Queue class to prevent race conditions.
    """

    def __init__(self, queue: SyncManager, lock: SyncManager) -> None:
        """
        Parameters
        ----------
        queue: SyncManager instance responsible for Queue instances.
        lock: SyncManager instance responsible for Lock instances.
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
        item: Function args and kwargs.

        Raises
        ------
        ValueError: Queue is empty.
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
    """
    Wrapper around the multiprocessing SharedMemory class to facilitate management of shared memory.
    """

    SUPPORTED = Union[str, int, float]

    def __init__(self, memory: SharedMemoryManager, lock: SyncManager, item: Type[SUPPORTED]) -> None:
        """
        Parameters
        ----------
        memory: SharedMemoryManager instance responsible for SharedMemory instances.
        lock: SyncManager instance responsible for Lock instances.
        item: Type of the shared object to allocate shared memory for.
        """
        sizes = {str: 128, int: 4, float: 8}
        reads = {str: self._read_str, int: self._read_int, float: self._read_float}
        writes = {str: self._write_str, int: self._write_int, float: self._write_float}

        self._type = item
        self._size = sizes[self._type]
        self._read = reads[self._type]
        self._write = writes[self._type]
        self._memory = memory.SharedMemory(size=self._size)
        self._lock = lock.Lock()
        self._encoding = "utf-8"

    def _read_str(self) -> str:
        """
        Read a string located in the shared memory.

        Returns
        -------
        item: String from the shared memory.
        """
        with self._lock:
            memory = SharedMemory(name=self._memory.name, create=False)
            item = memory.buf.tobytes().decode(encoding=self._encoding).rstrip("\x00")
            memory.close()

        return item

    def _read_int(self) -> int:
        """
        Read an integer located in the shared memory.

        Returns
        -------
        item: Integer from the shared memory.
        """
        with self._lock:
            memory = SharedMemory(name=self._memory.name, create=False)
            item = memory.buf[0]
            memory.close()

        return item

    def _read_float(self) -> float:
        """
        Read a float located in the shared memory.

        Returns
        -------
        item: Float from the shared memory.
        """
        with self._lock:
            memory = SharedMemory(name=self._memory.name, create=False)
            item = memory.buf.tobytes().decode(encoding=self._encoding).rstrip("\x00")
            memory.close()

        return float.fromhex(item)

    def _write_str(self, item: str) -> None:
        """
        Write a string to the shared memory.

        Parameters
        ----------
        item: String to write to the shared memory.
        """
        i_bytes = item.encode(encoding=self._encoding)
        i_length = len(i_bytes)

        with self._lock:
            shared_error = SharedMemory(name=self._memory.name, create=False)
            shared_error.buf[:i_length] = i_bytes[:i_length]
            shared_error.close()

    def _write_int(self, item: int) -> None:
        """
        Write an integer to the shared memory.

        Parameters
        ----------
        item: Integer to write to the shared memory.
        """
        with self._lock:
            memory = SharedMemory(name=self._memory.name, create=False)
            memory.buf[0] = item
            memory.close()

    def _write_float(self, item: float) -> None:
        """
        Write a float to the shared memory.

        Parameters
        ----------
        item: Float to write to the shared memory.
        """
        i_bytes = item.hex().encode(encoding=self._encoding)
        i_length = len(i_bytes)

        with self._lock:
            shared_error = SharedMemory(name=self._memory.name, create=False)
            shared_error.buf[:i_length] = i_bytes[:i_length]
            shared_error.close()

    def read(self) -> SUPPORTED:
        """
        Read the shared memory allocated to this Memory instance and return its value.

        Returns
        -------
        item: Decoded bytes from the shared memory as a Python object.
        """
        return self._read()

    def write(self, item: SUPPORTED) -> None:
        """
        Write the given item to the shared memory allocated to this Memory instance.

        Parameters
        ----------
        item: Python object to write to the shared memory.
        """
        self._write(item=item)


class ManagerShared:
    """
    Wrapper around the multiprocessing manager classes to facilitate management of shared memory, locks, queues, and
    processes.
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
        Request a lock which can be shared between processes.

        Returns
        -------
        lock: SyncManager Lock instance which can be shared between processes.
        """
        return self._manager_locks.Lock()

    def request_queue(self) -> Queue:
        """
        Request a queue which can be shared between processes.

        Returns
        -------
        queue: Queue instance which can be shared between processes.
        """
        return Queue(queue=self._manager_queues, lock=self._manager_locks)

    def request_memory(self, type_: Type[Memory.SUPPORTED]) -> Memory:
        """
        Request memory which can be shared between processes.

        Returns
        -------
        memory: Memory instance which can be shared between processes.
        """
        return Memory(memory=self._manager_memory, lock=self._manager_locks, item=type_)
