# ml_wave/models.py

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif


def get_pipelines():
    """Return dictionary of model pipelines for training."""

    pipelines = {
        'lda_to_knn': Pipeline([
            ('lda', LinearDiscriminantAnalysis(n_components=4)),
            ('knn', KNeighborsClassifier(n_neighbors=7))
        ]),

        'lda_to_logreg': Pipeline([
            ('lda', LinearDiscriminantAnalysis(n_components=4)),
            ('logreg', LogisticRegression(max_iter=1000))
        ]),

        'select_lda_knn': Pipeline([
            ('select', SelectKBest(f_classif, k=20)),
            ('lda', LinearDiscriminantAnalysis(n_components=4)),
            ('knn', KNeighborsClassifier(n_neighbors=35))
        ]),

        'direct_knn': Pipeline([
            ('knn', KNeighborsClassifier(n_neighbors=9))
        ]),

        'shrinkage_lda_knn': Pipeline([
            ('lda', LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')),
            ('knn', KNeighborsClassifier(n_neighbors=7))
        ])
    }

    return pipelines
