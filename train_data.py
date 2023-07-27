import csv

from labybrain import config


def generate_data(config_dict: dict):
    """
    Generates data for training the neural network.
    """
    cfg_copy = config_dict.copy()
    cfg_copy.pop("randomness")
    # add result
    cfg_copy["result"] = 0

    # построить все возможные комбинации
    # 1. сгенерировать все возможные комбинации из 0 и 1, result = 0
    # 2. записать их в файл

    with open('train_data.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cfg_copy.keys())
        writer.writeheader()
        # 1. сгенерировать все возможные комбинации из 0 и 1
        for i in range(2 ** (len(cfg_copy) - 1)):
            # 2. записать их в файл
            cfg_copy["result"] = 0
            for j in range(len(cfg_copy)):
                cfg_copy[list(cfg_copy.keys())[j]] = (i >> j) & 1
            writer.writerow(cfg_copy)


if __name__ == "__main__":
    generate_data(config)
