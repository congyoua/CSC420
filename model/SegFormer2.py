from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def get_model():
    id2label = {int(line.split()[0][:-1]): line.split()[1] for line in open("data/labels.txt")}
    label2id = {v: k for k, v in id2label.items()}
    return SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=7,
                                                            id2label=id2label, label2id=label2id)
