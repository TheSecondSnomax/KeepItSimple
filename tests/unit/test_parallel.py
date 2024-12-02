"""
This module contains the unit tests for the parallel module.
"""
from src.kiss.parallel import ManagerShared

from unittest import TestCase
from multiprocessing import Lock, Process
from time import sleep, time


def helper_test_request_lock_target(lock: Lock) -> None:
    with lock:
        sleep(1)


def helper_test_request_lock(processes: int, lock: Lock) -> float:
    processes = [Process(target=helper_test_request_lock_target, args=(lock,)) for _ in range(processes)]
    start = time()

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    if any(process.exitcode for process in processes):
        raise ChildProcessError

    stop = time()
    return stop - start


class TestParallel(TestCase):

    def test_request_lock_1(self) -> None:
        """
        Test if a lock can be requested, and block processes until it is released.
        """
        manager = ManagerShared()
        lock = manager.request_lock()
        processes = 2
        elapsed = helper_test_request_lock(processes=processes, lock=lock)

        self.assertTrue(expr=elapsed >= processes)

    def test_request_lock_2(self) -> None:
        """
        Test if a lock is garbage collected when the manager is destroyed.
        """
        manager = ManagerShared()
        lock = manager.request_lock()
        processes = 2
        del manager

        with self.assertRaises(expected_exception=ChildProcessError):
            helper_test_request_lock(processes=processes, lock=lock)
