import logging

def get_logger(name:str = __name__):
    """
    Returns a logger instance with the specified name.
    
    Args:
        name (str): The name of the logger. Defaults to the module's name.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logging.basicConfig(
        level =logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" ,
    )
   
    return logging.getLogger(name)