
"""with open("scripts/WebScraper.py", "r") as f:
    WebScraper = f.read()

exec(WebScraper)

with open("scripts/DataCleaner.py", "r") as f:
    DataCleaner = f.read()

exec(DataCleaner)


with open("scripts/Analyzer.py", "r") as f:
    Analyzer = f.read()

exec(Analyzer)


with open("scripts/Modeller.py", "r") as f:
    Modeller = f.read()

exec(Modeller)

with open("scripts/Predictor.py", "r") as f:
    Predictor = f.read()

exec(Predictor)
"""
from scripts.processor import Processor
from scripts.modeller import ProcessTrain
from functions.utils import create_df_from_pkl, create_pkl_from_df

def main():

    """input_filepath = "./new_colleagues.csv"
    names = read_names_from_csv(input_filepath)         # use function (in utils folder) to read names from file into a list
    names_random = random.sample(names, len(names))     # to go through the list random: (import random module at top of code)"""

    """df = open_pkl_as_dataframe(default_path = r'data\after_datacleaning.pkl')
    df_prep = Preprocessor(df)                            # make object of class Preprocessor()
    df_prep.outlier_handling_numerical()
    df_prep.outlier_handling_categorical()
    print(type(df_prep.df))
    save_dataframe_to_pkl(df_prep.df, file_path = r'data\after_analysis2.pkl') # Give dataframe to save, and path to file"""
    
    # Start of the process module: calling the methods in the class Processor
    df = create_df_from_pkl(default_path = r'data\after_analysis2.pkl')# Fill your path to file
    df_processor = Processor(df)
    print(type(df_processor.df))
    df_processor.df.info()
    df_processor.get_province()
    df_processor.assign_city_based_on_proximity_multiple_radii()
    X,y=df_processor.create_Xy()
    df_processor.ordinal_encoding(X)
    df_processor.onehot_encoding(X)

    best_pipeline = .models_linear(X,y)
    # Save the best pipeline with joblib
    joblib.dump(best_pipeline, 'best_model_pipeline.joblib')
    #df_prep.organize(names_random)                      # enter the number of tables via pop up (see Class Openspace, method .organize())
    #run_seatscout.find()
    #run_seatscout.display()                                       # display the different tables and their occupants in a nice and readable way in the terminal.
    #run_seatscout.store()                                         # store/save the repartitioned seat assignments to a new file (output.csv)

if __name__ == "__main__":
    main()