''
'训练O网络'

import net
from train_01 import train

if __name__ == '__main__':
    net = net.ONet()

    trainer = train.Trainer(net, r'E:\MTCNN\June\cebela_01\48',
                           'onet.pt' )
    trainer.train()