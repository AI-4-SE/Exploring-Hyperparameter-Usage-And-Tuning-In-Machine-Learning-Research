import json

sklearn_file_path = "../../data/library_data/sklearn_data.json"
tf_file_path = "../../data/library_data/tensorflow_data.json"
torch_file_path = "../../data/library_data/torch_data.json"


def get_library_statistics(library_file_path) -> None:
    with open(library_file_path, "r", encoding="utf-8") as src:
        data = json.load(src)

    total_api_call = 0
    total_params = 0

    for item in data:
        name = item["name"]
        if name[0].isupper():
            total_api_call += 1
            params = item["params"]
            total_params += len(params)

    return total_api_call, total_params


if __name__ == "__main__":
    sklearn_api_calls, sklearn_params = get_library_statistics(sklearn_file_path)
    tf_api_calls, tf_params = get_library_statistics(tf_file_path)
    torch_api_calls, torch_params = get_library_statistics(torch_file_path)
    
    print("Scikit-Learn API Calls and Params: ", sklearn_api_calls, sklearn_params)
    print("TensorFlow API Calls and Params: ", tf_api_calls, tf_params)
    print("PyTorch API Calls and Params: ", torch_api_calls, torch_params)