from generator import Generator
from helper import Helper

if __name__ == "__main__":
    helper = Helper()
    helper.read_dataset("images/prepared-set/")

    gen = Generator(helper, ".augmented/")
    gen.generate()
