import json

def get_tf_optimizer():
    
    optimizer = []

    with open("modules/tensorflow_default_values.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    # tensorflow.keras.optimizer
    # tensorflow.keras.optimizer.experimental
    # tensorflow.keras.optimizers.schedules
    # tensorflow.keras.dtensor.experimental.optimizers
    # tensorflow.keras.optimizers.legacy

    for item in data:
        #if "tensorflow.keras.optimizer" in item["full_name"]:
        #    parts = item["full_name"].split(".")
        #    if len(parts) == 4 and parts[-1][0].isupper(): 
        #        optimizer.append(item)
            
        #if "tensorflow.keras.dtensor.experimental.optimizers" in item["full_name"]:
        #    optimizer.append(item)

        #if "tensorflow.keras.optimizers.legacy" in item["full_name"]:
        #    optimizer.append(item)

        if "tensorflow.keras.optimizers.experimental" in item["full_name"]:
            optimizer.append(item)

    with open("modules/tensorflow_optimizer.json", "w", encoding="utf-8") as dest:
        json.dump(optimizer, dest, sort_keys=True, indent=4)
        

def get_pytorch_optimizer():
    
    optimizer = []

    with open("modules/torch_default_values.json", "r", encoding="utf-8") as src:
        data = json.load(src)

    # torch.optim

    for item in data:
        if "torch.optim" in item["full_name"]:
            parts = item["full_name"].split(".")
            if len(parts) == 3: 
                optimizer.append(item)


    with open("modules/torch_optimizer.json", "w", encoding="utf-8") as dest:
        json.dump(optimizer, dest, sort_keys=True, indent=4)
        

if __name__ == "__main__":
    #print_estimators()
    get_tf_optimizer()
    #get_pytorch_optimizer()
