sklearn_file = "../data/sklearn/sklearn_sample_url_final.csv"
tf_file = "../data/tensorflow/tensorflow_samples_url_final.csv"
torch_file = "../data/pytorch/pytorch_samples_url_final.csv"

new_files = "../data/5000/ml_samples_url_5000.csv"

files = [new_files]


def main():
    urls = set()

    for file_name in files:
        with open(file_name, "r", encoding="utf-8") as src:
            for line in src.readlines():
                if "tar_filename" in line or "repo_url" in line:
                    continue
                else:
                    url = line.split(",")[-1].strip()
                    urls.add(url)

    print(len(urls))
    print(urls)

    
if __name__ == "__main__":
    main()