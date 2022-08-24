import time


class TimeIt:
    def __init__(self, func_name):
        self.start = None
        self.end = None
        self.func_name = func_name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        dur = (self.end - self.start) * 1000  # to ms
        print(f"Function {self.func_name} ran in: {dur:.3f} ms")
