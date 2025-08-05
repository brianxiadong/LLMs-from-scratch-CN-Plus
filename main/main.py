import re
if __name__ == '__main__':
    with open("../ch02/01_main-chapter-code/the-verdict.txt", "r") as f:
        data = f.read()

    print(len(data))
    print(data[:99])

