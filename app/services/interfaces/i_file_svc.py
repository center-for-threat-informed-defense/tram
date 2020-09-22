import abc


class FileServiceInterface(abc.ABC):

    @abc.abstractmethod
    def save_file(self, filename, content, target_dir):
        pass

    @abc.abstractmethod
    def read_file(self, name, location):
        """
        Open a file and read the contents
        :param name:
        :param location:
        :return: a tuple (file_path, contents)
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def walk_file_path(path, target):
        pass

    @staticmethod
    @abc.abstractmethod
    def save_multipart_file_upload(request, target_dir):
        pass
