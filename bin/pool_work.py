import subprocess as sp
import os
import signal
import time
from typing import List, Optional


class ProcessPool:
    def __init__(self, jobs: List[str], count: Optional[int] = None):
        self.jobs = jobs
        self.done = []
        self.failed = []
        self.active_processes = []
        self.count_limit = count or os.cpu_count()

    def spawn_next(self) -> bool:
        if len(self.active_processes) >= self.count_limit:
            return False

        if len(self.jobs) == 0:
            return False

        print(f"Spawning {self.jobs[0]}...")
        self.active_processes.append(
            sp.Popen(self.jobs[0], shell=True, preexec_fn=os.setsid)
        )
        self.jobs = self.jobs[1:]
        return True

    def running(self) -> bool:
        return len(self.active_processes) != 0 or len(self.jobs) != 0

    def __call__(self) -> None:
        print("Spawning processes...")

        for _ in range(self.count_limit):
            self.spawn_next()

        print("Entering loop...")

        while self.running():
            try:
                done_processes = [
                    p for p in self.active_processes if p.poll() is not None
                ]
                self.active_processes = [
                    p for p in self.active_processes if p not in done_processes
                ]

                for p in done_processes:
                    if p.returncode != 0:
                        self.failed.append(p.args)
                    else:
                        self.done.append(p.args)
                    self.spawn_next()

                time.sleep(1)
                if done_processes:
                    print(f"Total done after this iteration: {len(self.done)}")
                    print(f"Total left after this iteration: {len(self.jobs)}")
            except KeyboardInterrupt:
                for p in self.active_processes:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                    self.failed.append(p.args)
                self.active_processes = []
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("TO_DO", help="file with commands to run")
    parser.add_argument(
        "DONE", help="file to which write done jobs", default="done", nargs="?"
    )
    parser.add_argument(
        "UN_DONE", help="file to which write not done jobs", default="undone", nargs="?"
    )
    parser.add_argument(
        "--cpu-count", "-x", help="cpu count", default=os.cpu_count(), type=int
    )
    args = parser.parse_args()

    with open(args.TO_DO) as f:
        todos = [x.strip() for x in f.read().strip().splitlines()]

    pool = ProcessPool(todos, args.cpu_count)
    pool()

    with open(args.DONE, "w") as f:
        f.write("\n".join(pool.done))

    with open(args.UN_DONE, "w") as f:
        f.write("\n".join(pool.jobs + pool.failed))
