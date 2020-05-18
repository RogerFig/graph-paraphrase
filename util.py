
def get_labels(input_f):
    labels = []
    with open(input_f, 'r') as f:
        for line in f.readlines():
            labels.append(float(line))
    return labels
