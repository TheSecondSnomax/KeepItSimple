"""
This module contains the classes and functions used for parallelism.
"""
from multiprocessing.managers import SharedMemoryManager, SyncManager


class ManagerShared:
    """
    Wrapper around the multiprocessing manager classes to facilitate management of neuron shared memory and locks.
    The lifetime of the manager must be at least as long as the neurons,
    otherwise the neurons will try to access deallocated memory.
    """

    def __init__(self):
        """
        Notes
        -----
        pylint "consider-using-with" error disabled:
            SharedMemoryManager has to remain running after initialisation for the reason mentioned above.
        """
        # pylint: disable=consider-using-with
        self._m_manager = SharedMemoryManager()
        self._m_manager.start()

        self._l_manager = SyncManager()
        self._l_manager.start()
        # pylint: enable=consider-using-with

        self._queue = self._l_manager.Queue()
        self._memory = []
        self._locks = []
        self._processes = []

    def __del__(self):
        self._m_manager.shutdown()
        self._l_manager.shutdown()
