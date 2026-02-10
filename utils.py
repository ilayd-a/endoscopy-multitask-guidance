import numpy as np

def set_class_values(all_classes, classes_to_train):
    return [all_classes.index(cls.lower()) for cls in classes_to_train]

def get_label_mask(mask, class_values, label_colors_list):
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask

