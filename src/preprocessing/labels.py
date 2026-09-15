from sklearn.preprocessing import MultiLabelBinarizer

def parse_label_ids(label_string):
    return [
        int(label_id.strip())
        for label_id in label_string.split(",")
    ]

def encode_multilabels(label_lists, num_labels):
    mlb = MultiLabelBinarizer(
        classes=list(range(num_labels))
    )

    encoded = mlb.fit_transform(label_lists)

    return encoded, mlb