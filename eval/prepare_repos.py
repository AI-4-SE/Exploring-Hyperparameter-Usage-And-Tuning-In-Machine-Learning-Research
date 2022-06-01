def main():
    urls = []

    with open("../data/sklearn/sklearn_sample_url_final.csv", "r", encoding="utf-8") as src:
        for line in src.readlines():
            
            url = line.split(",")[-1].strip()
            urls.append(url)


        urls = urls[1:]    
    print(len(urls))
if __name__ == "__main__":
    main()