from threading import Lock
import uuid
import random

class NumGenerator():

    ctx_id = 1
    lock = Lock()
    flowno = 1024

    @staticmethod
    def flowno(self):
        with self.lock:
            self.flowno += 1
            return self.flowno
    
    @staticmethod
    def ctx_id(self):
        with self.lock:
            self.ctx_id += 1
            if self.ctx_id == 0:
                self.ctx_id_+= 1
            return self.ctx_id

    @staticmethod
    def session(self):
        return str(uuid.uuid4())

    @staticmethod
    def randno(self):
        return random.randint(-1e10, 1e10)