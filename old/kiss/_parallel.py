"""
This module contains the classes and functions used for parallelism.
"""

from inspect import stack
from multiprocessing import Process, Queue, cpu_count
from multiprocessing.managers import SharedMemoryManager, SyncManager
from multiprocessing.shared_memory import SharedMemory
from threading import Lock
from typing import Any, Callable, Optional

from numpy import dtype, single


def get_worker_status(memory_status: str, lock: Lock) -> int:
    """
    Get the status of a worker process from the shared memory.

    Parameters
    ----------
    memory_status: str
        Shared memory reference to the worker process status.
    lock: Lock
        SyncManager lock instance dedicated to the worker process.

    Returns
    -------
    status: int
        Returns True if the worker process is idle, False otherwise.
    """
    with lock:
        shared = SharedMemory(name=memory_status, create=False)
        status = shared.buf[0]
        shared.close()

    return status


def set_worker_status(status: int, memory_status: str, lock: Lock) -> None:
    """
    Set the status of a worker process in the shared memory.

    Parameters
    ----------
    status: int
        Value of the worker process status as a boolean.
    memory_status: str
        Shared memory reference to the worker process status.
    lock: Lock
        SyncManager lock instance dedicated to the worker process.
    """
    with lock:
        shared = SharedMemory(name=memory_status, create=False)
        shared.buf[0] = status
        shared.close()


def get_worker_error(memory_error: str, lock: Lock, encoding: str = "utf-8") -> str:
    """
    Get the error string of a worker process from the shared memory.

    Parameters
    ----------
    memory_error: str
        Shared memory reference to the worker process error string.
    lock: Lock
        SyncManager lock instance dedicated to the worker process.
    encoding: str
        Decode the error bytes from the shared memory using the given encoding.

    Returns
    -------
    error: str
        Returns the error string if the worker process failed, otherwise an empty string.
    """
    with lock:
        shared_error = SharedMemory(name=memory_error, create=False)
        error = shared_error.buf[:256].tobytes().decode(encoding).rstrip("\x00")
        shared_error.close()

    return error


def set_worker_error(error: str, memory_error: str, size: int, lock: Lock, encoding: str = "utf-8") -> None:
    """
    Set the error string of a worker process in the shared memory.

    Parameters
    ----------
    error: str
        Worker process error string.
    memory_error: str
        Shared memory reference to the worker process error string.
    size: int
        Shared memory size in bytes.
    lock: Lock
        SyncManager lock instance dedicated to the worker process.
    encoding: str
        Encode the error string to bytes in the shared memory using the given encoding.
    """
    e_bytes = error.encode(encoding)
    e_length = size if len(e_bytes) > 256 else len(e_bytes)

    with lock:
        shared_error = SharedMemory(name=memory_error, create=False)
        shared_error.buf[:e_length] = e_bytes[:e_length]
        shared_error.close()


def worker(queue: Queue, memory_status: str, memory_error: str, lock: Lock, size: int = 256) -> None:
    """
    Process work items from the given queue until either this process is terminated, or a work item fails.

    Parameters
    ----------
    queue: Queue
        SyncManager queue instance from which to get work items.
    memory_status: str
        Shared memory reference to the worker process status.
    memory_error: str
        Shared memory reference to the worker process error string.
    lock: Lock
        SyncManager lock instance dedicated to the worker process.
    size: int
        Shared memory size in bytes for the worker process error string.
    """
    status = 1

    while True:
        if not queue.empty():
            method, parameters = queue.get(timeout=1)
            status = 0
            set_worker_status(status=status, memory_status=memory_status, lock=lock)

            try:
                method(**parameters)
            except Exception as e:
                set_worker_error(error=str(e), memory_error=memory_error, size=size, lock=lock)
                raise e
        else:
            if not status:
                status = 1
                set_worker_status(status=status, memory_status=memory_status, lock=lock)


def guard() -> bool:
    """
    This function can be used to guard against endless recursion caused by starting multiprocess.Process instances.

    Returns
    -------
    main: bool
        Return True if the current process is the main process, False otherwise.
    """
    for item in stack():
        name = item.frame.f_globals["__name__"]

        if "multiprocessing" in name:
            return False

    return True


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

        self._dtype = single
        self._queue = self._l_manager.Queue()
        self._memory = []
        self._locks = []
        self._processes = []

    def request_memory(self, samples: int, weights: int) -> dict:
        """
        Request shared memory for the NeuronShared instance, with sizes based on the given samples and weights.

        Parameters
        ----------
        samples: int
            Number of samples to base the shared memory allocation on.
        weights:
            Number of weights to base the shared memory allocation on.

        Returns
        -------
        memory: dict
            Dictionary containing the shared memory references.
        """
        x = self._m_manager.SharedMemory(size=samples * dtype(dtype=self._dtype).itemsize)
        y = self._m_manager.SharedMemory(size=samples * dtype(dtype=self._dtype).itemsize)
        y_prime = self._m_manager.SharedMemory(size=samples * dtype(dtype=self._dtype).itemsize)
        wts = self._m_manager.SharedMemory(size=weights * dtype(dtype=self._dtype).itemsize)
        bias = self._m_manager.SharedMemory(size=dtype(dtype=self._dtype).itemsize)

        internal = [x, y, y_prime, wts, bias]
        external = {"x": x.name, "y": y.name, "y_prime": y_prime.name, "weights": wts.name, "bias": bias.name}

        self._memory.append(internal)
        return external

    def request_lock(self) -> Lock:
        """
        Request shared memory for the NeuronShared instance.

        Returns
        -------
        lock: Lock
            SyncManager lock instance which can be shared across processes.
        """
        internal = self._l_manager.Lock()

        self._locks.append(internal)
        return internal

    def request_workers(self, workers: Optional[int] = None) -> None:
        """
        Request workers which process work items from the managed FIFO queue.

        Parameters
        ----------
        workers: int
            Number of workers to start. This number will always be reduced to the number of CPU cores - 1, if higher.
        """
        workers = cpu_count() - 1 if workers is None else workers

        for _ in range(workers):
            size = 256
            status = self._m_manager.SharedMemory(size=4)
            error = self._m_manager.SharedMemory(size=size)
            lock = self._l_manager.Lock()

            set_worker_status(status=1, memory_status=status.name, lock=lock)
            set_worker_error(error="", memory_error=error.name, lock=lock, size=size)

            process = Process(target=worker, args=(self._queue, status.name, error.name, lock))
            process.start()

            self._memory.append(status)
            self._memory.append(error)
            self._locks.append(lock)
            self._processes.append({"process": process, "status": status.name, "error": error.name, "lock": lock})

    def stop_workers(self) -> None:
        """
        Stop the requested workers.
        """
        for process in self._processes:
            process["process"].terminate()

        self._processes = []

    def add_work(self, function: Callable, parameters: dict[str, Any]) -> None:
        """
        Add work items to the managed FIFO queue.

        Parameters
        ----------
        function: Callable
            Function to be targeted by the worker.
        parameters: dict
            Parameters to call the targeted function with as key-value pairs.
        """
        self._queue.put(item=[function, parameters])

    def get_status(self) -> bool:
        """
        Get the status of the workers and the work items.

        Returns
        -------
        status: bool
            Returns True if the workers are idle and the queue is empty.
            Returns False if the workers are not idle or the queue is not empty.

        Raises
        ------
        Exception
            A worker has failed. The exception is raised with the error message from the worker process.
        """
        statuses = []

        for process in self._processes:
            error = get_worker_error(memory_error=process["error"], lock=process["lock"])

            # pylint: disable=broad-exception-raised
            if error:
                raise Exception(error)
            # pylint: enable=broad-exception-raised

            status = get_worker_status(memory_status=process["status"], lock=process["lock"])
            statuses.append(status)

        if not self._queue.empty():
            return False

        return all(statuses)

    def __del__(self):
        for process in self._processes:
            process["process"].terminate()

        self._m_manager.shutdown()
        self._l_manager.shutdown()
