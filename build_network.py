import resnet
class Net(object):
    def __init__(self, base_network_name, is_training):
        self.base_network_name = base_network_name
        self.is_training = is_training
    def build_base_network(self, input_img_batch):
        return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)