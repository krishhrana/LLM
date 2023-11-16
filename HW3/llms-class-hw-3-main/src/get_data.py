def fetch_data():
    from datasets import load_dataset
    data_files = {
        "train": "en.noclean/c4-train.0000[0-1]-of-07168.json.gz",
    }
    dataset = load_dataset("allenai/c4", data_files=data_files, split="train")
    val_split = dataset.select(range(int(len(dataset) * 0.95), len(dataset)))
    train_split = dataset.select(range(0, int(len(dataset) * 0.95)))

    return train_split, val_split #dataset[:len(dataset) * 0.95], dataset[len(dataset) * 0.95:]

    
