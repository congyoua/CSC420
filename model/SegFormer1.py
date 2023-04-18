from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def get_model():
    id2label = {0: 'background', 1: 'human'}
    label2id = {v: k for k, v in id2label.items()}
    return SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2,
                                                            id2label=id2label, label2id=label2id)
