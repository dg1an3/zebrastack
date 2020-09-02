class ImageOp:
    def __init__(self):        
        self.hits = 0
        self.miss = 0

    def get_cache_value(self, img):
        import hashlib
        hasher = hashlib.md5()
        hasher.update(str(img.shape).encode('utf-8'))
        hasher.update(img.data)
        for_hash = hasher.hexdigest()    
        if for_hash in cache:
            self.hits = self.hits + 1
            return for_hash, cache[for_hash]
        self.miss = self.miss + 1
        return for_hash, None
    
class HistoOp(ImageOp):    
    def __call__(self, img):
        print('HistoOp {}'.format(img))