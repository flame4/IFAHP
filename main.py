from framework import Framework
from config import conf_file, case_cnt


def main():
    for i in range(1, case_cnt):
        try:
            case_name = "case%d" % i
            f = Framework(conf_file, case_name)
            f.build()
            if not f.after_build():
                raise RuntimeError("Framework build Fail!")
            print(f)
        except Exception as e:
            print(e.__repr__() + ", Init Failed!")
            break


if __name__ == '__main__':
    main()
