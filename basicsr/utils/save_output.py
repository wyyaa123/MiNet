class SaveOutput:
    """
    the class aims to save the feature map of a block/networlk output
    """
    def __init__(self):
        self.outputs = []
        
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
        
    def clear(self):
        self.outputs = []