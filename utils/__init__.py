import os
try:
    from resource import getrusage, RUSAGE_CHILDREN,RUSAGE_SELF

    def get_memory_mb():
        res = {
            "self": getrusage(RUSAGE_SELF).ru_maxrss/1024,
            "children": getrusage(RUSAGE_CHILDREN).ru_maxrss/1024,
            "total": getrusage(RUSAGE_SELF).ru_maxrss/1024 + getrusage(RUSAGE_CHILDREN).ru_maxrss/1024
        }
        return res
except:
    def get_memory_mb():
        return {
            "self": -1,
            "children": -1,
            "total": -1
        }

def create_if_not_exists(path: str) -> None:
    """
    Creates the specified folder if it does not exist.
    :param path: the complete path of the folder to be created
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


