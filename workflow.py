class WorkflowHandler():
    """
    Class to contain all workflows, so that they can be called via the scripts
    """

    def __init__(self):
        """to be completed"""
        return
    
    def preprocessor_workflow(self):
        self.df = open_pkl_as_dataframe(self.input_filepath)
        df.outlier_handling_numerical()
        df.outlier_handling_categorical()
        print(type(df_prep.df))
        save_dataframe_to_pkl(df_prep.df, file_path = r'data\after_analysis2.pkl') # Give dataframe to save, and path to file"""
        return  