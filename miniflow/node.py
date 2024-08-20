

class Node(object):
    def __init__(self):
        self.inputs = []
        self.op = None
        self.cong_attr = None
        self.name = ''
    def __add__(self,other):
        pass
    def __mul__(self,other):
        pass 
    
    __radd__ = __add__
    __rmul__ = __mul__
    
    def __str__(self):
        return self.name
    
    
