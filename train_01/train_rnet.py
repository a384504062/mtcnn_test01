''
'训练R网络'

import net
from train_01 import train

if __name__ == '__main__':
    net = net.RNet()

    trainer = train.Trainer(net, r'E:\MTCNN\June\cebela_01\24',
                           'rnet.pt' )
    trainer.train()