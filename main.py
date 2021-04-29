from generator import Generator
from methods import *

if __name__ == "__main__":
    helper = Helper()
    # helper.read_image("images/prepared-set/Chun", "Chun\n")
    helper.read_frame("images/prepared-set/")

    gen = Generator(helper, ".augmented/")
    gen.generate()
