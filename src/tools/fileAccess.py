# -*- coding: utf8 -*-
import contextlib
import gzip
import logging
import os
import random


def files_iterator(path, shuffle=False, type_filter="", recursive=False):
    """
    Iterate files found in the path.
    If the path is a file return only this one, if the path is a directory iterate files inside (and in its sub-folders
    if recursive option is True).

    :param path: str : The path explore
    :param shuffle: bool : If files need to be shuffled before return
    :param type_filter: str : The type of file to iterate. Silently ignore file which doesn't match with the type
    filter. Empty filter means no type filter
    :param recursive: If true explore sub-folder recursively, otherwise only explore the current path
    :return: Generator[str] : A generator of file path
    """
    if not os.path.exists(path):
        raise ValueError("Path not found: " + path)

    # If the path lead directly to a file...
    if os.path.isfile(path):
        # No file type return any file
        if not type_filter:
            yield path
        # File type checking
        # If file doesn't match the type ignore silently
        elif os.path.splitext(path)[1][1:] == type_filter:
            yield path

    # If the path lead to a directory iterate files inside
    elif os.path.isdir(path):
        listDir = os.listdir(path)
        if shuffle:
            random.shuffle(listDir)
        for item in listDir:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path) and recursive:
                # Replace with "yield from" with python 3
                for sub_files in files_iterator(item_path, shuffle, type_filter, recursive):
                    yield sub_files
            elif os.path.isfile(item_path):
                # No file type return any file
                if not type_filter:
                    yield item_path
                # File type checking
                # If file doesn't match the type ignore silently
                elif os.path.splitext(item_path)[1][1:] == type_filter:
                    yield item_path


def lines_iterator_file(path, file_type="auto", default_file_type="txt"):
    """
    Iterate lines of a text file

    :param path: str : The path to file location
    :param file_type: str : The file type, "txt" and "gz" are currently supported. "auto" guess the file type from the
    file extension
    :param default_file_type: str : The default file type if auto fail to guess it
    :return: Generator[str] : A generator of line
    """

    # Try to guess the file type from the extension (with no dot)
    if file_type == "auto":
        file_type = os.path.splitext(path)[1][1:]
        if not file_type:
            if default_file_type:
                file_type = default_file_type
            else:
                raise RuntimeError("Fail to auto detect file extension. Explicit file type seems necessary. PATH=" +
                                   path)

    logging.debug("Open file: " + path)
    ignoreFile = False
    if file_type == "txt" or file_type == "log":
        f = open(path, 'rt')
    elif file_type == "gz":
        f = gzip.open(path, 'rt')
    elif file_type == "json":
        # Ignore this file
        ignoreFile = True
    else:
        raise NotImplementedError("File type '" + file_type + "' not supported")

    if not ignoreFile:
        with contextlib.closing(f):
            for line in f:
                yield line


def lines_iterator_directory(path, shuffle=False, file_type="auto", type_filter="", recursive=False, default_file_type="txt"):
    """
    Iterate each lines for each files found in the path.

    :param path: str : The path explore, should be a directory
    :param file_type: str The file type, "txt" and "gz" are currently supported. "auto" guess the file type from the
    file extension
    :param type_filter: str : The type of file to iterate. Silently ignore file which doesn't match with the type
    filter. Empty filter means no type filter
    :param recursive: If true explore sub-folder recursively, otherwise only explore the current path
    :param default_file_type: str : The default file type if auto fail to guess it
    :return: Generator[str] A line generator
    """

    if not os.path.isdir(path):
        raise ValueError("No directory found at: " + path)

    for file_path in files_iterator(path, shuffle, type_filter, recursive):
        for line in lines_iterator_file(file_path, file_type, default_file_type):
            yield line
