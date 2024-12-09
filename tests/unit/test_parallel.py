"""
This module contains the unit tests and their helper functions for the parallel module.
"""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from multiprocessing import Process
from multiprocessing.managers import SyncManager
from sys import exit
from threading import Lock
from time import sleep, time
from typing import Callable
from unittest import TestCase

from src.kiss.parallel import ManagerShared, Queue


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


def helper_request_queue(queue: Queue) -> None:
    """
    Target function for the "request_queue" method tests.
    """
    while True:
        try:
            queue.get()
            sleep(1)
        except ValueError:
            exit(0)


class TestParallel(TestCase):
    """
    This test case contains the unit tests for the parallel module.
    """

    def test_queue_put(self) -> None:
        """
        Test if an item can be put in the queue.
        """
        with SyncManager() as manager:
            queue = Queue(queue=manager, lock=manager)
            queue.put(1, 2, a=3, b=4)
            args, kwargs = queue._queue.get()

        self.assertEqual(first=(1, 2), second=args)
        self.assertEqual(first={"a": 3, "b": 4}, second=kwargs)

    def test_queue_get(self) -> None:
        """
        Test if an item can be retrieved from the queue.
        """
        with SyncManager() as manager:
            queue = Queue(queue=manager, lock=manager)
            queue._queue.put(((1, 2), {"a": 3, "b": 4}))
            args, kwargs = queue.get()

        self.assertEqual(first=(1, 2), second=args)
        self.assertEqual(first={"a": 3, "b": 4}, second=kwargs)

    def test_queue_empty(self) -> None:
        """
        Test if an error is raised when the queue is empty.
        """
        with SyncManager() as manager:
            queue = Queue(queue=manager, lock=manager)

            with self.assertRaises(expected_exception=ValueError):
                queue.get()

    def test_request_lock_1(self) -> None:
        """
        Test if a lock can be requested, and blocks other processes until it is released.
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

    def test_request_queue_1(self) -> None:
        """
        Test if a queue can be requested, and can be used across processes.
        """
        manager = ManagerShared()
        queue = manager.request_queue()

        for i in range(2):
            queue.put(i)

        processes = 2
        elapsed = helper_start_testers(processes=processes, target=helper_request_queue, queue=queue)

        self.assertTrue(expr=elapsed < processes)

    def test_request_queue_2(self) -> None:
        """
        Test if a queue is garbage collected when the manager is destroyed.
        """
        manager = ManagerShared()
        queue = manager.request_queue()

        for i in range(2):
            queue.put(i)

        processes = 1
        del manager

        with self.assertRaises(expected_exception=ChildProcessError):
            helper_start_testers(processes=processes, target=helper_request_queue, queue=queue)
