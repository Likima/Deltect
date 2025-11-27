# CHROMOSOMES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]  # Multiple chromosomes
CHROMOSOMES = ["2"]
MAX_VARIANTS_PER_CHR = 20000
TEST_SIZE = 0.1
CV_FOLDS = 5  # Reduced from 10 for faster training with larger dataset
SAVE_OUTPUTS = True
REFERENCE_FASTA = "hs37d5.fa"
BALANCE_CLASSES = True