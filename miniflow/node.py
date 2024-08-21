

class Node(object):
    def __init__(self):
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ''
    
    def __str__(self):
        return self.name
    
 
