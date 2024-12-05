from multiprocessing import Lock, Process, Queue
from multiprocessing.shared_memory import SharedMemory
from threading import Thread
from time import sleep, time
from unittest import TestCase, main
from unittest.mock import patch

from numpy import dtype, ndarray, single
from pytest import mark
from snoscience._parallel import (
    ManagerShared,
    get_worker_error,
    get_worker_status,
    guard,
    set_worker_error,
    set_worker_status,
    worker,
)


def helper_lock(lock: Lock, seconds: int) -> None:
    with lock:
        sleep(seconds)


def helper_worker_1(parameter: int) -> int:
    return parameter


def helper_worker_2(parameter: int) -> int:
    raise ValueError(parameter)


def helper_guard() -> None:
    if not guard():
        raise ValueError()


class HelperMemory:

    def __init__(self, size: int):
        self.memory = SharedMemory(create=True, size=size)
        self.name = self.memory.name

    def __del__(self):
        self.memory.close()
        self.memory.unlink()


class HelperQueue:

    def __init__(self):
        self.queue = Queue()

    def __del__(self):
        self.queue.close()
        self.queue.join_thread()


class TestParallel(TestCase):

    def test_get_worker_status(self):
        """
        Test if integer can be retrieved from the shared memory as a worker status.
        """
        status_true = 1
        lock = Lock()

        memory = HelperMemory(size=4)
        memory.memory.buf[0] = status_true

        status_calc = get_worker_status(memory_status=memory.name, lock=lock)
        self.assertEqual(first=status_true, second=status_calc)

    def test_set_worker_status(self):
        """
        Test if integer can be set in the shared memory as a worker status.
        """
        status_true = 1
        lock = Lock()

        memory = HelperMemory(size=4)
        set_worker_status(status=status_true, memory_status=memory.name, lock=lock)

        memory_calc = SharedMemory(name=memory.name, create=False)
        status_calc = memory_calc.buf[0]
        memory_calc.close()

        self.assertEqual(first=status_true, second=status_calc)

    def test_get_worker_error(self):
        """
        Test if string can be retrieved from the shared memory as a worker error.
        """
        error_true = "an error occurred"
        error_len = len(error_true)
        error_bytes = error_true.encode(encoding="utf-8")
        lock = Lock()

        memory = HelperMemory(size=error_len)
        memory.memory.buf[:error_len] = error_bytes

        error_calc = get_worker_error(memory_error=memory.name, lock=lock)
        self.assertEqual(first=error_true, second=error_calc)

    def test_set_worker_error(self):
        """
        Test if string can be set in the shared memory as a worker error.
        """
        error_true = "an error occurred"
        error_len = len(error_true)
        lock = Lock()

        memory = HelperMemory(size=error_len)
        set_worker_error(error=error_true, memory_error=memory.name, lock=lock, size=256)

        memory_calc = SharedMemory(name=memory.name, create=False)
        error_calc = memory_calc.buf.tobytes().decode("utf-8").rstrip("\x00")
        memory_calc.close()

        self.assertEqual(first=error_true, second=error_calc)

    @mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @patch(target="kiss._parallel.set_worker_status")
    @patch(target="kiss._parallel.set_worker_error")
    def test_worker_1(self, mock_error, mock_status):
        """
        Test if "worker" function gets work from queue and updates its status.
        """
        lock = Lock()
        queue = HelperQueue()
        queue.queue.put(obj=[helper_worker_1, {"parameter": None}])

        thread = Thread(target=worker, args=(queue.queue, None, None, lock))
        thread.daemon = True
        thread.start()

        self.assertTrue(expr=thread.is_alive())
        thread.join(timeout=1)
        self.assertTrue(expr=thread.is_alive())

        self.assertEqual(first=2, second=mock_status.call_count)
        self.assertEqual(first=0, second=mock_error.call_count)

    @mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
    @patch(target="kiss._parallel.set_worker_status")
    @patch(target="kiss._parallel.set_worker_error")
    def test_worker_2(self, mock_error, mock_status):
        """
        Test if "worker" function updates its error when a work item failed.
        """
        lock = Lock()
        queue = HelperQueue()
        queue.queue.put(obj=[helper_worker_2, {"parameter": None}])

        thread = Thread(target=worker, args=(queue.queue, None, None, lock))
        thread.daemon = True
        thread.start()

        self.assertTrue(expr=thread.is_alive())
        thread.join(timeout=1)
        self.assertFalse(expr=thread.is_alive())

        self.assertEqual(first=1, second=mock_status.call_count)
        self.assertEqual(first=1, second=mock_error.call_count)

    def test_guard_1(self):
        """
        Test if "guard" function returns True if the process is the main process.
        """
        self.assertTrue(expr=guard())

    def test_guard_2(self):
        """
        Test if "guard" function returns False if the process is not the main process.
        """
        process = Process(target=helper_guard)
        process.start()
        process.join()

        self.assertEqual(first=1, second=process.exitcode)

    def test_request_memory_1(self):
        """
        Test if shared memory can be requested.
        """
        manager = ManagerShared()
        memory = manager.request_memory(samples=2, weights=4)

        def test_size(item: str, size: int):
            shared = SharedMemory(name=memory[item], create=False)
            buffer = ndarray(shape=(size, 1), dtype=single, buffer=shared.buf)

            size_true = size * dtype(dtype=single).itemsize
            size_calc = buffer.size * buffer.itemsize
            shared.close()

            self.assertEqual(first=size_true, second=size_calc)

        # Test memory depending on samples.
        test_size(item="x", size=2)
        test_size(item="y", size=2)
        test_size(item="y_prime", size=2)

        # Test memory assigned to weights.
        test_size(item="weights", size=4)

        # Test memory assigned to bias.
        test_size(item="bias", size=1)

    def test_request_memory_2(self):
        """
        Test if shared memory is deallocated when the manager is destroyed.
        """
        manager = ManagerShared()
        memory = manager.request_memory(samples=2, weights=4)

        del manager

        with self.assertRaises(expected_exception=FileNotFoundError):
            SharedMemory(name=memory["x"], create=False)
        with self.assertRaises(expected_exception=FileNotFoundError):
            SharedMemory(name=memory["y"], create=False)
        with self.assertRaises(expected_exception=FileNotFoundError):
            SharedMemory(name=memory["y_prime"], create=False)
        with self.assertRaises(expected_exception=FileNotFoundError):
            SharedMemory(name=memory["weights"], create=False)
        with self.assertRaises(expected_exception=FileNotFoundError):
            SharedMemory(name=memory["bias"], create=False)

    def test_request_lock(self):
        """
        Test if a lock can be requested, and block processes until it is released.
        """
        manager = ManagerShared()
        lock = manager.request_lock()
        seconds = 1
        processes = [Process(target=helper_lock, args=(lock, seconds)) for _ in range(2)]
        start = time()

        for process in processes:
            process.start()
        for process in processes:
            process.join()

        stop = time()
        self.assertTrue((stop - start) >= (seconds * len(processes)))

    def test_request_workers_1(self):
        """
        Test if worker processes can be requested and stopped.
        """
        manager = ManagerShared()
        manager.request_workers(workers=2)
        processes = [process["process"] for process in manager._processes]

        for process in processes:
            self.assertTrue(expr=process.is_alive())

        manager.stop_workers()

        for process in processes:
            self.assertFalse(expr=process.is_alive())

    def test_request_workers_2(self):
        """
        Test if worker processes can be requested and are stopped when the manager is destroyed.
        """
        manager = ManagerShared()
        manager.request_workers(workers=2)
        processes = [process["process"] for process in manager._processes]

        for process in processes:
            self.assertTrue(expr=process.is_alive())

        del manager

        for process in processes:
            self.assertFalse(expr=process.is_alive())

    def test_add_work(self):
        """
        Test if work items can be added to the managed queue.
        """
        manager = ManagerShared()
        size = 3

        for i in range(size):
            manager.add_work(function=helper_worker_1, parameters={"parameter": i})

        self.assertEqual(first=size, second=manager._queue.qsize())

        for i in range(size):
            func, params = manager._queue.get()

            self.assertEqual(first=helper_worker_1, second=func)
            self.assertEqual(first={"parameter": i}, second=params)

        self.assertTrue(expr=manager._queue.empty())

    def test_get_status_1(self):
        """
        Test if "get_status" method returns True without workers and an empty queue.
        """
        manager = ManagerShared()

        self.assertTrue(expr=manager.get_status())

    def test_get_status_2(self):
        """
        Test if "get_status" method returns True with idle workers and an empty queue.
        """
        manager = ManagerShared()
        manager.request_workers(workers=2)

        self.assertTrue(expr=manager.get_status())

    def test_get_status_3(self):
        """
        Test if "get_status" method returns False without workers and a filled queue.
        """
        manager = ManagerShared()
        manager.add_work(function=helper_worker_1, parameters={"parameter": None})

        self.assertFalse(expr=manager.get_status())

    @patch(target="kiss._parallel.get_worker_status")
    @patch(target="kiss._parallel.get_worker_error")
    def test_get_status_4(self, mock_error, _):
        """
        Test if "get_status" method raises an error when a worker failed.
        """
        mock_error.return_value = "an error occurred"

        manager = ManagerShared()
        manager._processes = [{"process": None, "error": None, "lock": None}]

        with self.assertRaises(Exception) as error:
            manager.get_status()
        try:
            self.assertEqual(first=mock_error.return_value, second=str(error.exception))
        finally:
            manager._processes = []


if __name__ == "__main__":
    main()
