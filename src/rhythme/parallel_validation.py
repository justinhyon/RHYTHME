from rhythme import __version__

try:
    from .Configuration import run_rhythme
except ImportError:
    from Configuration import run_rhythme
import concurrent
import multiprocessing as mp
import sys
import os
import logging
import warnings
from itertools import repeat
from os import listdir
from os.path import isfile, join
import time


SENTINEL = 'DONE'
def block_print(func):
    def block(*args, **kwargs):
        # Redirect stdout
        sys.stdout = open(os.devnull, 'w')  # Redirecting to /dev/null in Unix-like systems

        # Call the function
        result = func(*args, **kwargs)

        # Restore stdout
        sys.stdout = sys.__stdout__  # Restore original stdout

        return result

    return block

# @block_print
def rhythme_no_print(config, file):
    run_rhythme(config, file)

def worker(id, q, config_file):
    print(f'{id}:: Worker running', flush=True)
    # better to use the while True with SENTINEL
    # other methods such as checking 'q.empty()' may be unreliable.
    while True:
        file = q.get(timeout=3)
        if file == SENTINEL:
            q.task_done()
            break
        print(f'{id}::Working on {file}', flush=True)
        blockAllPrint()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rhythme_no_print(config_file, file)
        # enablePrint()
        # run_rhythme(config_file, file)
        # enablePrint()
        print(f'{id}::Finished {file}', flush=True)
        q.task_done()

        print(f'{id}::Sleeping. Item: {file}', flush=True)

        time.sleep(0.1)
    print(
        f'We reached the end.', flush=True)

def main(files, threads, config_file):

    print('running main')
    # Send thirty task requests to the worker.
    with mp.Manager() as manager:
        q = manager.Queue()

        for item in files:
            q.put(item)

        # adding 4 sentinel values at the end, 1 for each process.
        for _ in range(threads):
            q.put(SENTINEL)

        # Confirm that queue is filled
        print(f'Approx queue size: {q.qsize()}')
        id = 0

        # start process pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            executor.map(worker, range(mp.cpu_count()), repeat(q), repeat(config_file))

        print('working')
        # Block until all tasks are done.
        q.join()
        print('All work completed')

# Disable
def blockAllPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


if __name__ == "__main__":
    validation_path = "non-python/data_3a_play"
    config_file = "config_offline_3a_parallel.toml"

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    files = [validation_path + '/' + f for f in listdir(validation_path) if
             (isfile(join(validation_path, f)) and f.endswith(".set"))]



    files.append(files[0])  # temporary
    print('files to be run:')
    for f in files:
        print(f)

    main(files, mp.cpu_count(), config_file)
    # blockPrint()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
    #     map = executor.map(thread_function, range(mp.cpu_count()), repeat(config_file), files)
    #
    # for res in map:
    #     print(res)

