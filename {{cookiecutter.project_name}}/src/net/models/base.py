class BaseModel(object):
    def __init__(self, config) -> None:
        self.config = config
    
    def build(self):
        raise NotImplementedError