from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from Util import load_files
mb, not_mb = load_files()

from Util import load_mixed_fixed
train_csv = load_mixed_fixed()

from Util import load_giga_mixed_fixed
giga_train_csv = load_giga_mixed_fixed()


