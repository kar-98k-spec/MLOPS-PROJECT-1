from src.logger import get_logger
from src.custom_exception import CustomException
import sys


logger = get_logger(__name__)

def div_num(a,b):
    try:
        result = a/b
        logger.info("divides two numbers")
        return result
    except Exception as e:
        logger.error("Error occured")
        raise CustomException("cannot divide by Zero",sys)
    
if __name__ =="__main__":
    try:
       logger.info("starting program")
       ans = div_num(5,10)
       print(ans)
    except CustomException as ce:
       logger.error(str(ce))