import json

def main():
    with open("final_metadata.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    print("Length Data: ", len(data))
    

    sklearn_counter = 0
    tf_counter = 0
    torch_counter = 0

    sklearn_torch_counter = 0
    sklearn_tf_counter = 0
    tf_torch_counter = 0

    sklearn_torch_tf_counter = 0

    non_library_counter = 0

    for item in data:
        code_stats = item["code_stats"]
        imports = code_stats["imports"]
        
        if "sklearn" in imports:
            sklearn_counter += 1
        
        if "tensorflow" in imports:
            tf_counter += 1

        if "torch" in imports:
            torch_counter += 1

        if "sklearn" in imports and "tensorflow" in imports:
            sklearn_tf_counter += 1

        if "sklearn" in imports and "torch" in imports:
            sklearn_torch_counter += 1

        if "torch" in imports and "tensorflow" in imports:
            tf_torch_counter += 1

        if "sklearn" in imports and "tensorflow" in imports and "torch" in imports:
            sklearn_torch_tf_counter += 1

        if "sklearn" not in imports and "tensorflow" not in imports and "torch" not in imports:
            non_library_counter += 1

    print("Number of repos with sklearn: ", sklearn_counter)
    print("Number of repos with tf: ", tf_counter)
    print("Number of repos with pytorch: ", torch_counter)

    print("Number of repos with sklearn and pytorch: ", sklearn_torch_counter)
    print("Number of repos with sklearn and tf: ", sklearn_tf_counter)
    print("Number of repos with tf and pytorch: ", tf_torch_counter)
    print("Combination of two: ", sklearn_torch_counter + sklearn_tf_counter + tf_torch_counter)

    print("Number of repos with all three: ", sklearn_torch_tf_counter)
    print("Number of repos without any library: ", non_library_counter)


if __name__ == "__main__":
    main()