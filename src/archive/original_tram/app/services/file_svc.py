import os

from app.utility.base_world import BaseWorld


class FileService(BaseWorld):

    def __init__(self):
        self.log = self.add_service('file_svc', self)
        self.data_svc = self.get_service('data_svc')

    async def save_file(self, filename, content, target_dir):
        with open('%s' % os.path.join(target_dir, filename), 'wb') as f:
            f.write(content)

    async def read_file(self, name, location='data'):
        file_name = await self.walk_file_path(location=location, target=name)
        if file_name:
            with open(file_name, 'rb') as f:
                buf = f.read()
            return buf
        raise FileNotFoundError

    @staticmethod
    async def walk_file_path(location, target):
        for root, _, files in os.walk(location):
            if target in files:
                return os.path.join(root, target)
        return None

    async def save_multipart_file_upload(self, request, target_dir):
        try:
            reader = await request.multipart()
            while True:
                field = await reader.next()
                if not field:
                    break
                filename = field.filename
                await self.save_file(filename, bytes(await field.read()), target_dir)
                self.log.debug('Uploaded file %s/%s' % (target_dir, filename))
            return '%s%s' % (target_dir, filename)
        except Exception as e:
            self.log.debug('Exception uploading file: %s' % e)
