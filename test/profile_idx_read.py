from mnistdp import IdxFile
from line_profiler import LineProfiler


def do_profile(follow=[]):
    def inner(func):
        def profiled_func(*args, **kwargs):
            try:
                profiler = LineProfiler()
                profiler.add_function(func)
                for f in follow:
                    profiler.add_function(f)
                profiler.enable_by_count()
                return func(*args, **kwargs)
            finally:
                profiler.print_stats()

        return profiled_func

    return inner


@do_profile(follow=[IdxFile.generator, IdxFile._read_data_sample])
def main():
    tr_images_file = "/home/tylee/01_dataset/MNIST/train-images-idx3-ubyte"
    tr_images = IdxFile.from_file(tr_images_file)
    gen = tr_images.generator()
    i = 0
    try:
        while i < 20:
            image = gen.__next__()
            i += 1
    except StopIteration:
        print(f"Generator exhausted")


if __name__ == "__main__":
    main()
