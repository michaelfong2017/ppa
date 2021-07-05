# %%
import concurrent.futures as cf
from functools import wraps


"""
Usage example 1 (multithreading):

    @make_parallel()
    def double(x):
        print(threading.current_thread().getName())
        return x * 2

    if __name__ == "__main__":
        result = double([1, 2, 3, 4])
        print(result)

Usage example 2 (multithreading)
    
    @make_parallel(has_different_tasks=False)
    def multiply(x1, x2, x3, x4):
        print(threading.current_thread().getName())
        return x1 * x2 * x3 * x4

    if __name__ == "__main__":
        result = multiply(1, 2, 3, 4)
        print(result)

Usage example 3 (multithreading)

    @make_parallel(has_different_tasks=True, has_multiple_arguments=True)
    def multiply(x1, x2, x3, x4):
        print(threading.current_thread().getName())
        return x1 * x2 * x3 * x4

    if __name__ == "__main__":
        result = multiply([(1, 2, 3, 4), [1, 3, 5, 7]])
        print(result)

Usage example 4 (multiprocessing)

    import datetime
    def double(x):
        print(f"Process ID: {os.getpid()}")
        return x ** 5000000 * 0

    '''
    With multiprocessing
    '''
    start_time = datetime.datetime.now()
    mp_double = make_parallel(type="multiprocessing", max_workers=8)(double)
    result = mp_double([7, 7, 7, 7, 7, 7, 7, 7])
    print(result)
    print(f"Time elapsed: {datetime.datetime.now() - start_time}")

    '''
    Without multiprocessing
    '''
    start_time = datetime.datetime.now()
    result = [(i, double(i)) for i in [7, 7, 7, 7, 7, 7, 7, 7]]
    print(result)
    print(f"Time elapsed: {datetime.datetime.now() - start_time}")
"""


def make_parallel(
    type="multithreading",
    max_workers=None,
    has_different_tasks=True,
    has_multiple_arguments=False,
):
    """
    Decorator used to decorate any function which needs to use multithreading or multiprocessing.

    :param type:

        "multithreading": uses ThreadPoolExecutor

        "multiprocessing": uses ProcessPoolExecutor

        Default: "multithreading"

    :param max_workers:
    Is the number of parallel tasks to be executed at the same time.

        Default: None, which means as many worker processes will be created as the machine has processors.

    :param has_different_tasks:
    Different tasks mean tasks with different combinations of arguments.

        True: args[0] of the wrapped function is required to be an iterable.
        All other args and kwargs will be abandoned. Then, args[0]
        is iterated and each element is the packed / unpacked arguments of a task.
        e.g.
        Input: func(["url1", "url2", "url3"]).
        Executed: func("url1"), func("url2"), func("url3").

        False: args is directly the packed arguments for a single task.
        args is then unpacked, i.e. func(*args) is executed.
        e.g.
        Input: func("url1", "url2", "url3").
        Executed: func("url1", "url2", "url3").
        ---
        Input: func(["url1", "url2", "url3"]).
        Executed: func(["url1", "url2", "url3"]).

        Default: True, since it is more likely to execute different tasks in parallel,
        e.g. making requests to different urls.

    :param has_multiple_arguments:
    Take effect only when has_different_tasks is True.

        True: Unpack arguments of each task.
        e.g.
        Input: func([("a", 1), ["b", 2], ["c", 3], ["d", 4]]).
        Executed: func("a", 1), func("b", 2), func("c", 3), func("d", 4).

        False: Do not unpack arguments of each task.
        e.g.
        Input: func(["url1", "url2", "url3"]).
        Executed: func("url1"), func("url2"), func("url3").
        ---
        Input: func([("a", 1), ["b", 2], ["c", 3], ["d", 4]]).
        Executed: func(("a", 1)), func(["b", 2]), func(["c", 3]), func(["d", 4]).

        Default: False, since in the case of single argument, it is intuitive for the client not to pack the arguments.
        In the case of multiple arguments, the client can realize the need of this parameter and manually switch this on.
    """

    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if type == "multithreading":
                executor = cf.ThreadPoolExecutor(max_workers=max_workers)
            elif type == "multiprocessing":
                executor = cf.ProcessPoolExecutor(max_workers=max_workers)
            else:
                return None

            with executor:
                """
                except block is used to handle the case that func takes no arguments,
                i.e. func() is executed.
                """
                try:
                    if has_different_tasks:
                        tasks_args = args[0]
                        # Reach here if args[0] exists.
                        """
                        Use case 1:
                            Input: func(["url1", "url2", "url3"]).
                            Expected execution: func("url1"), func("url2"), func("url3").
                            Therefore, should NOT unpack each element.

                        Use case 2:
                            Input: func([("a", 1), ["b", 2], ["c", 3], ["d", 4]]).
                            Expected execution: func("a", 1), func("b", 2), func("c", 3), func("d", 4).
                            Therefore, should unpack each element.

                        Use case 3 (rare):
                            Input: func([("a", 1), ["b", 2], ["c", 3], ["d", 4]]).
                            Expected execution: func(("a", 1)), func(["b", 2]), func(["c", 3]), func(["d", 4]).
                            Therefore, should NOT unpack each element.

                        Use case 4 (very rare):
                            Input: func(["00", "01", "10", "11"])
                            Expected execution: func("0", "0"), func("0", "1"), func("1", "0"), func("1", "1").
                            Therefore, should unpack each element.

                        Use case 5 (very rare):
                            Input: func("1234").
                            Expected execution: func("1"), func("2"), func("3"), func("4").
                            Therefore, no difference to unpack each element or not.

                        Exception 1:
                            args[0] is not Iterable.
                            e.g. Input: func(123).
                            TypeError.

                        Exception 2:
                            Try to unpack an element that is not Iterable.
                            e.g. Input: func([1, 2, 3]), has_multiple_arguments=True.
                            TypeError.
                        """
                        try:
                            if has_multiple_arguments:
                                """
                                Unpack each element.
                                """
                                future_to_args = {
                                    executor.submit(func, *task_args, **kwargs): task_args
                                    for task_args in tasks_args
                                }
                            else:
                                """
                                Does not unpack each element.
                                """
                                future_to_args = {
                                    executor.submit(func, task_args, **kwargs): task_args
                                    for task_args in tasks_args
                                }

                        except TypeError as e:
                            if has_multiple_arguments:
                                print(
                                    f"""TypeError: {e}
Reason: 
It is known that args[0] is the first argument to your custom function "func" and args[0] exists.
You set has_different_tasks=True and has_multiple_arguments=True, which tries to unpack an element of args[0].
However, either args[0] is not Iterable or this element is not Iterable."""
                                )
                            else:
                                print(
                                    f"""TypeError: {e}
Reason: 
It is known that args[0] is the first argument to your custom function "func" and args[0] exists.
You set has_different_tasks=True, which tries to iterate args[0].
However, args[0] is not Iterable."""
                                )
                            return None

                    else:
                        """
                        Has a single task only. Therefore, args is directly the packed arguments for a single task
                        and args is then unpacked.
                        """
                        future_to_args = {executor.submit(func, *args, **kwargs): args}

                except IndexError as e:
                    """
                    In the case of no arguments, simply execute func.
                    i.e. func()
                    """
                    future_to_args = {executor.submit(func, **kwargs): None}

                result = []
                for future in cf.as_completed(future_to_args):
                    result.append((future_to_args[future], future.result()))

            if has_different_tasks:
                return result
            else:
                return result[0]

        return wrapped

    return wrapper

# %%
