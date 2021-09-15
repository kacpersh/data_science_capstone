import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from logging import root
from pathlib import Path
from typing import ContextManager, Union


def extension_extractor(file: Union[str, os.PathLike]) -> str:
    """Extracts and returns file extensions from a file

    :param file: str or :class:`os.PathLike` target to write
    :return: str representation of a file extensions
    """
    f_suffixes = Path(file).suffixes
    f_ext = "".join(f_suffixes)
    return f_ext


@contextmanager
def atomic_write(
    file: Union[str, os.PathLike], mode: str = "w",
        as_file: bool = True, **kwargs) -> ContextManager:
    """Write a file atomically

    :param file: str or :class:`os.PathLike` target to write
    :param mode: the mode in which the file is opened,
                defaults to "w" (writing in text mode)
    :param bool as_file:  if True, the yielded object is a :class:File.
        (eg, what you get with `open(...)`).  Otherwise, it will be the
        temporary file path string

    :param kwargs: anything else needed to open the file

    :raises: FileExistsError if target exists

    Example::

        with atomic_write("hello.txt") as f:
            f.write("world!")
    """
    if os.path.exists(file) is True:
        raise FileExistsError
    f_temp = tempfile.NamedTemporaryFile(
        dir=os.path.split(file)[0], suffix=extension_extractor(file), mode="w+"
    )
    if as_file is False:
        try:
            yield f_temp.name
        finally:
            f_temp.close()
    else:
        try:
            yield f_temp
        except Exception:
            root.info("Unexpected error:", sys.exc_info()[0])
            raise
        else:
            f_dest = open(file, mode, **kwargs)
            f_temp.seek(0)
            shutil.copyfileobj(f_temp, f_dest)
            f_dest.close()
        finally:
            f_temp.close()
