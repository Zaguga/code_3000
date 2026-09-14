# packages
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314

def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=100, #from 100 to 150 -- Made it worse, reverted back to 100
        max_depth=3, #from 8 to 4 -- to 3
        subsample=1,
        min_samples_leaf=2, #from 1 to 2 --back to 1 -- increasing this lowered it by .002 -- testing to 4 now.. (keeping other 3)
        # 4 raised it again, keep at 2, and decrease max dept to 2? -- Still not most optimal (.112)
        #current is most optimal at .111
        random_state=seed
    )
    model.fit(X, y)
    return model