"""Definition of Conditional Variables"""
import multiprocessing as mp
import ctypes


class MasterWorkersCV:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self._done_ctr = mp.Value(ctypes.c_int, num_workers)
        self._works_done = mp.Array(
            ctypes.c_bool, [True for _ in range(num_workers)])
        self._cv_master = mp.Condition(self._done_ctr.get_lock())
        self._cv_workers = [mp.Condition() for _ in range(num_workers)]

    def sync_round(self):
        with self._cv_master:
            self._done_ctr.get_obj().value = 0

            for wid in range(self.num_workers):
                self._works_done[wid] = False
                with self._cv_workers[wid]:
                    self._cv_workers[wid].notify()

            while not self._all_work_done():
                self._cv_master.wait()

    def wait_for_work(self, wid):
        """workers[wid] waits for work

        caller should not grap the worker CV;
        worker CV is released after the function returns
        """
        with self._cv_workers[wid]:
            while self._works_done[wid]:
                self._cv_workers[wid].wait()

    def work_done_maybe_notify_master(self, wid):
        """mark the work as done and notify master only when all work is done."""
        self._works_done[wid] = True

        with self._cv_master:
            self._done_ctr.get_obj().value += 1

            if self._all_work_done():
                self._cv_master.notify()

    def _all_work_done(self):
        """Caller should be grabbing the master CV."""
        return self._done_ctr.get_obj().value == self.num_workers
