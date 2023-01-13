import glob

def main():

    files = glob.glob("../data/statistics/*")
    print("Length files: ", len(files))


if __name__ == "__main__":
    main()