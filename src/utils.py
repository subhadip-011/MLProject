# utils is for common functionlaties which is used in entire project
import os
import sys # for exception handling

import pandas as pd
import numpy as np
import dill # this library is used for save the pickle file 
from src.exception  import CustomException

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)  # save the pickle file 
            
    except Exception as e:
        raise  CustomException(e,sys)