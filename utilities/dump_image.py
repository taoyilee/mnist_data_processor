from mnistdp import IdxFile
import os
from PIL import Image


def dump_image(idx_file_name, output_dir):
    idx_file = IdxFile.from_file(idx_file_name)
    _, idx_file_fullname = os.path.split(idx_file_name)
    idx_file_basename, _ = os.path.splitext(idx_file_fullname)
    output_dir = os.path.join(output_dir, f"{idx_file_basename}")
    os.makedirs(output_dir, exist_ok=True)

    image_generator = idx_file.generator()
    i = 0
    try:
        while True:
            image = Image.fromarray(image_generator.__next__())
            output_image = os.path.join(output_dir, f"{idx_file_basename}_{i:05d}.png")
            print(f"Writing out {output_image}")
            image.save(output_image)
            i += 1
    except StopIteration:
        print(f"Generator exhausted")


if __name__ == "__main__":
    dump_image("/home/tylee/01_dataset/MNIST/train-images-idx3-ubyte", "images")
