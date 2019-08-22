from pymongo import MongoClient
from typing import Any, Optional, Tuple

# Custom imports
from ..utils.constants import DB_NAME

__all__ = [
    "init_collection",
    "is_running",
    "MongoError",
    "MongoWorker"
    ]


class MongoError(Exception):
    """Custom exception for MongoDB errors.
    """
    pass


class MongoWorker:
    """Context manager for connecting to database and collection.
    
    Parameters
    ----------
    host : str
        IP address of MongoDB instance
    
    port : int
        Port of MongoDB instance

    collection : str
        Name of MongoDB collection
    """
    def __init__(self, host: str, port: int, collection: Optional[str] = None) -> None:
        self.host: str                 = host
        self.port: int                 = port
        self.collection: Optional[str] = collection


    def __enter__(self) -> Any:
        """Get MongoDB collection or database with context manager.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        db : MongoClient
            MongoDB database or collection connection instance
        """
        # Create connection and get collection
        self.client: MongoClient = MongoClient(host=self.host, port=self.port)
        db: Any                  = self.client[DB_NAME]
        return db[self.collection] if self.collection else db
           

    def __exit__(self, *args: Any) -> None:
        self.client.close()


def is_running(host: str, port: int) -> Tuple[bool, Any]:
    """Checks if MongoDB is running at specified host and port.
    
    Parameters
    ----------
    host : str
        IP address of MongoDB instance
    
    port : int
        Port of MongoDB instance

    Returns
    -------
    status : bool
        True if MongoDB is running, otherwise False
    
    msg : str or Exception
        Message for connection status
    """
    try:
        client: MongoClient = MongoClient(host=host, port=port, serverSelectionTimeoutMS=1)
        info: str           = client.server_info()
        return (True, "MongoDB is running at %s:%s" % (host, port))
    except Exception as e:
        return (False, e)


def insert_init_record(collection: MongoClient, computer_name: str) -> None:
    """Initializes collection with initial record.
    
    Parameters
    ----------
    collection : MongoClient
        MongoDB collection instance

    computer_name : str
        Unique name of computer
    
    Returns
    -------
    None
    """
    collection.insert({
            "computer_name" : computer_name,
            "message"       : "initialized collection",
        })


def init_collection(
    host: str, 
    port: int, 
    collection: str, 
    overwrite: bool, 
    computer_name: str
    ) -> Tuple[bool, Any]:
    """Initializes MongoDB collection.
    
    Parameters
    ----------
    host : str
        IP address of MongoDB instance
    
    port : int
        Port of MongoDB instance

    collection : str
        Name of MongoDB collection

    overwrite : bool
        Whether to overwrite database and collection
    
    computer_name : str
        Unique name of computer
    
    Returns
    -------
    status : bool
        True if MongoDB is running, otherwise False
    
    msg : str or Exception
        Message for connection status
    """
    try:
        msg: str = ""
        with MongoWorker(host, port, collection=None) as db:
            if collection not in db.collection_names():
                db_collection: MongoClient = db[collection]
                insert_init_record(db_collection, computer_name)
            else:
                if overwrite:
                    db[collection].drop()
                    db_collection = db[collection]
                    insert_init_record(db_collection, computer_name)
                else:
                    msg = "collection %s already created" % collection

        # Default message to successfully created new collection
        if not msg: 
            msg = "successfully created new collection %s" % collection
        return (True, msg)
    
    except Exception as e:
        return (False, e)