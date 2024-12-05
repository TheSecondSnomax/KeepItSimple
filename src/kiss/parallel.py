"""
This module contains the classes and functions used for parallelism.
"""

from multiprocessing import Lock
from multiprocessing.managers import SharedMemoryManager, SyncManager


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

        self._queue = self._manager_locks.Queue()

    def __del__(self) -> None:
        self._manager_memory.shutdown()
        self._manager_locks.shutdown()

    def request_lock(self) -> Lock:
        """
        Request a lock which can be shared across processes.

        Returns
        -------
        lock: Lock
            SyncManager lock instance which can be shared across processes.
        """
        return self._manager_locks.Lock()
