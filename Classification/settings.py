FILENAME_GENUINE = r'C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\genuine1_MCYT.csv'
FILENAME_FORGERY = r'C:\Users\andra\Documents\Egyetem\Mesteri\Disszentacio\Project\forgery1_MCYT.csv'


#Dataset related settings
NR_OF_USERS = 100
NR_OF_GENUINE_SIGS_OF_USER = 25
NR_OF_FORGERY_SIGS_OF_USER = 25

#Random Forest
N_ESTIMATORS_RF = 100
NUM_FOLDS_RF = 10

#OneClassSVM
TRAINING_SAMPLES_SVM = 15
GENUINE_SIGNATURES_SVM = 10 
FORGERY_SIGNATURES_SVM = 25

#Measurment settings -> if true, skilled forgery scenario will be executed, if false, random forgery
SKILLED_FORGERY = True
