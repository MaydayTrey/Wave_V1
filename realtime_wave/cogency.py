import numpy as np
#This should import our knn model in real time predictions and confidences

'''The goal of this is to pull the kNN probabilities, LogReg probabilities, 
and generate a confidence score, then to check if the confidences are aligned. 
If they both anticipate the same label, we will check the confidences and then fuse them.
If that confidence is above a threshold we output that as the outputted label.

IF: The most likely label is in dispute, we output UNCERTAIN, and if the confidence when fused
is below the threshold we output UNCERTAIN as well.

lda_to_knn -> Probabilities and max confidence = knn_pred, knn_conf
lda_to_logreg -> Probabilities and max confidence = logreg_pred, logreg_conf

IF knn_pred == logreg_pred:
    label output is equal to this mutually agreed label
    and cogency_conf = (knn_conf + logreg_conf) / 2
ELSE:
    label is in an uncertain state
    and cogency_conf is = ((knn_conf + logreg_conf) / 2) * dispute_weight

IF congency_conf > threshold:
    label output is the agreed upon label
ELSE:
    label is in an uncertain state
'''
def cogency_fusion(knn_proba, logreg_proba, dispute_weight=0.5, strict_agreement=False):
    """
    Fuses KNN and LogReg probabilities by averaging.

    Args:
        knn_proba: (n_samples, n_classes) from KNN
        logreg_proba: (n_samples, n_classes) from LogReg
        dispute_weight: Multiply confidence by this when models disagree
        strict_agreement: If True, return -1 when models disagree

    Returns:
        labels: np.ndarray of int class indices (0-5, or -1)
        confidences: np.ndarray of float confidence scores
    """
    fused_proba = (knn_proba + logreg_proba) / 2.0

    knn_labels = np.argmax(knn_proba, axis=1)
    logreg_labels = np.argmax(logreg_proba, axis=1)
    fused_labels = np.argmax(fused_proba, axis=1)

    confidences = np.max(fused_proba, axis=1)

    disagreement = knn_labels != logreg_labels

    if strict_agreement:
        fused_labels = fused_labels.astype(int)
        fused_labels[disagreement] = -1
        confidences[disagreement] = 0.0
    else:
        confidences[disagreement] *= dispute_weight

    return fused_labels.astype(int), confidences
