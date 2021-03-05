import copy
import glob
import os.path
import pickle
import shutil

from app.utility.base_world import BaseWorld


class DataService(BaseWorld):

    def __init__(self):
        self.log = self.add_service('data_svc', self)
        self.schema = dict(search=[], reports=[])
        self.ram = copy.deepcopy(self.schema)

    @staticmethod
    async def destroy():
        if os.path.exists('data/object_store'):
            os.remove('data/object_store')
        for d in ['data/reports']:
            for f in glob.glob('%s/*' % d):
                if not f.startswith('.'):
                    try:
                        os.remove(f)
                    except IsADirectoryError:
                        shutil.rmtree(f)

    async def save_state(self):
        await self.get_service('file_svc').save_file('object_store', pickle.dumps(self.ram), 'data')

    async def restore_state(self):
        if os.path.exists('data/object_store'):
            store = await self.get_service('file_svc').read_file('object_store', 'data')
            ram = pickle.loads(store)
            for key in ram.keys():
                self.ram[key] = []
                for c_object in ram[key]:
                    await self.store(c_object)
            self.log.debug('Restored data from persistent storage')

    async def store(self, c_object):
        try:
            return c_object.store(self.ram)
        except Exception as e:
            self.log.error('[!] can only store first-class objects: %s' % e)

    async def locate(self, object_name, match=None):
        try:
            return [obj for obj in self.ram[object_name] if obj.match(match)]
        except Exception as e:
            self.log.error('[!] LOCATE: %s' % e)

    async def remove(self, object_name, match):
        try:
            self.ram[object_name][:] = [obj for obj in self.ram[object_name] if not obj.match(match)]
        except Exception as e:
            self.log.error('[!] REMOVE: %s' % e)
