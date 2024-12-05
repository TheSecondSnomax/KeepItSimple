"""
This module contains the classes and functions used for parallelism.
"""

from multiprocessing.managers import SharedMemoryManager, SyncManager
from threading import Lock


class Queue:

    def __init__(self, manager: SyncManager):
        self._queue = manager.Queue()

    def add(self, *args, **kwargs):
        self._queue.put(item=[args, kwargs])


class ManagerShared:
    """
    Wrapper around multiprocessing manager classes to facilitate management of shared memory, locks, queues,
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
        return Queue(self._manager_queues)
