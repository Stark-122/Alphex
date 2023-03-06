def open_file(dataset, mode):
    with open(dataset, mode, encoding='utf-8') as f:
        data = f.read()
        return data