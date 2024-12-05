"""
This module contains the unit tests and their helper functions for the parallel module.
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from multiprocessing import Lock, Process
from sys import exit
from time import sleep, time
from typing import Callable
from unittest import TestCase

from src.kiss.parallel import ManagerShared


def helper_start_testers(processes: int, target: Callable, **kwargs) -> float:
    """
    Start child processes testing a target function.
    """
    processes = [Process(target=helper_tester, args=(target,), kwargs=kwargs) for _ in range(processes)]
    start = time()

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    if any(process.exitcode for process in processes):
        raise ChildProcessError

    stop = time()
    return stop - start


def helper_tester(target: Callable, **kwargs) -> None:
    """
    Wrapper around the child process target function to redirect stdout and stderr.
    """
    stream = StringIO()
    code = 1

    with redirect_stdout(stream), redirect_stderr(stream):
        try:
            target(**kwargs)
            code = 0
        except Exception as e:
            print(e)

    exit(code)


def helper_request_lock(lock: Lock) -> None:
    """
    Target function for the "request_lock" method tests.
    """
    with lock:
        sleep(1)


class TestParallel(TestCase):
    """
    This test case contains the unit tests for the parallel module.
    """

    def test_request_lock_1(self) -> None:
        """
        Test if a lock can be requested, and blocks processes until it is released.
        """
        manager = ManagerShared()
        lock = manager.request_lock()
        processes = 2
        elapsed = helper_start_testers(processes=processes, target=helper_request_lock, lock=lock)

        self.assertTrue(expr=elapsed >= processes)

    def test_request_lock_2(self) -> None:
        """
        Test if a lock is garbage collected when the manager is destroyed.
        """
        manager = ManagerShared()
        lock = manager.request_lock()
        processes = 1
        del manager

        with self.assertRaises(expected_exception=ChildProcessError):
            helper_start_testers(processes=processes, target=helper_request_lock, lock=lock)
