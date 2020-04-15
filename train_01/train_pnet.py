''
'训练P网络'

import net
from train_01 import train

if __name__ == '__main__':
    net = net.PNet()

    trainer = train.Trainer(net, r'E:\MTCNN\June\cebela_01\12',
                           'pnet.pt' )
    trainer.train()