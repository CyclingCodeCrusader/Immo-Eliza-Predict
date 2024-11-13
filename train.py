from scripts.preprocessor import Preprocessor
from scripts.processor import Processor
from scripts.modeller import Modeller
from functions.utils import create_df_from_pkl, create_pkl_from_df
import joblib

# Start of the preprocess module: calling the methods in the class Preprocessor
#preprocessor = Preprocessor()                # make object of class Preprocessor()
#preprocessor.run_workflow()                  # run workflow through the methods in the Preprocessor class                          

# Start of the process module: calling the methods in the class Processor
processor = Processor()                       # make object of class Processor()
df = processor.train_workflow()             # run workflow through the methods in the Processor class   

# Start of the modeller module: calling the methods in the class Modeller
modeller = Modeller(df)                      # make object of class Modeller()
modeller.train_workflow()

#df_prep.organize(names_random)                      # enter the number of tables via pop up (see Class Openspace, method .organize())
#run_seatscout.find()
#run_seatscout.display()                                       # display the different tables and their occupants in a nice and readable way in the terminal.
#run_seatscout.store()                                         # store/save the repartitioned seat assignments to a new file (output.csv)
