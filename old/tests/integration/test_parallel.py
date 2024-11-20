from os import getcwd, mkdir, path
from shutil import rmtree
from time import sleep, time
from unittest import TestCase, main

from snoscience._parallel import ManagerShared

MESSAGE = "Parameters do not match."


def worker(name: str, content: str):
    with open(file=f"{getcwd()}/files/{name}.txt", mode="w") as file:
        file.write(content)


def worker_fail(name: str, content: str):
    if name != content:
        raise ValueError(MESSAGE)


class TestParallel(TestCase):

    @classmethod
    def setUpClass(cls):
        if path.exists("./files"):
            rmtree(path="./files")

        mkdir(path="./files")

    @classmethod
    def tearDownClass(cls):
        if path.exists("./files"):
            rmtree(path="./files")

    def test_workers_1(self):
        """
        Test if a worker signals its finished status to the manager, check if the work has been done after.
        """
        manager = ManagerShared()
        manager.request_workers(workers=2)
        timeout = 10
        items = []

        for i in range(12):
            items.append({"name": f"name_{i}", "content": f"content_{i}"})

        for item in items:
            manager._queue.put(obj=[worker, item])

        start = time()
        elapsed = 0

        while not manager.get_status() and elapsed < timeout:
            elapsed = time() - start
            sleep(0.1)

        self.assertTrue(expr=manager._queue.empty())
        self.assertTrue(expr=manager.get_status())

        for item in items:
            with open(file=f"./files/{item['name']}.txt", mode="r") as file:
                content = file.read()
                self.assertEqual(first=item["content"], second=content)

    def test_workers_2(self):
        """
        Test if a worker signals its waiting status to the manager.
        """
        manager = ManagerShared()
        manager.request_workers(workers=2)

        self.assertTrue(expr=manager.get_status())

    def test_workers_3(self):
        """
        Test if a worker signals an error to the manager, and if it is raised in the main process.
        """
        manager = ManagerShared()
        manager.request_workers(workers=2)

        items = [{"name": "name", "content": "content"}, {"name": "name", "content": "content"}]

        for item in items:
            manager._queue.put(obj=[worker_fail, item])

        sleep(1)

        with self.assertRaises(Exception) as error:
            manager.get_status()

        self.assertEqual(first=MESSAGE, second=str(error.exception))


if __name__ == "__main__":
    main()
