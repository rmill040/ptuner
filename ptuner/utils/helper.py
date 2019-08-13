import logging
import random
import socket
import time

_LOGGER = logging.getLogger(__name__)

__all__ = [
    "get_hostname",
    "get_ip_address",
    "countdown"
    ]


def get_hostname() -> str:
    """Get unique hostname for computer.
    
    Parameters
    ----------
    None

    Returns
    -------
    name : str
        Unique name of host
    """
    try:
        return socket.gethostname()
    except Exception as e:
        name: str = "hostname_" + str(random.randint(0, 1000))
        msg: str  = "unable to identify hostname because %s, setting hostname to %s" % \
                (e, name)
        _LOGGER.warn(msg)
        return name


def get_ip_address() -> str:
    """Get IP address for computer.
        
    Parameters
    ----------
    None

    Returns
    -------
    ip_address : str
        IP address of computer
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = str(s.getsockname()[0])
        s.close()
        return ip_address
    except:
        return "unknown"


def countdown(message: str, t: int) -> None:
    """Provides a countdown printed to the console.

    Parameters
    ----------
    message : str
        Message to print
    
    t : int
        Time in seconds to start timer

    Returns
    -------
    None
    """
    while t:
        mins, secs = divmod(t, 60)
        timeformat: str = '[info] {} in {:02d}:{:02d}'.format(message, mins, secs)
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1