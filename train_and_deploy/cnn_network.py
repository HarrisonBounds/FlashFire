
# Neural Network architecture definitions

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init

class simpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv64 = nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2))
        self.conv128 = nn.Conv2d(16, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv256 = nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(1, 1))

        #SPATIAL DIMENSION FORMULA (Assume no padding)
        #(Input height - kernel height) / stride) + 1

        self.fc1 = nn.Linear(32*70*70, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        

    def forward(self, x):               
        x = self.relu(self.conv64(x))  
        x = self.relu(self.conv128(x))  
        x = self.relu(self.conv256(x))  

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class moderateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv48 = nn.Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1))
        self.conv64 = nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1))

        #SPATIAL DIMENSION FORMULA (Assume no padding)
        #(Input height - kernel height) / (stride + 1)


        self.fc1 = nn.Linear(64*43*43, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):         
        x = self.relu(self.conv24(x))        
        x = self.relu(self.conv32(x))  
        x = self.relu(self.conv48(x))  
        x = self.relu(self.conv64(x))  

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class megaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv16 = nn.Conv2d(3, 16, kernel_size=(5, 5), stride=(2, 2))
        self.conv24 = nn.Conv2d(16, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(3, 3), stride=(2, 2))
        self.conv48 = nn.Conv2d(32, 48, kernel_size=(2, 2), stride=(1, 1))
        self.conv64 = nn.Conv2d(48, 64, kernel_size=(1, 1), stride=(1, 1))


        #SPATIAL DIMENSION FORMULA (Assume no padding)
        #(Input height - kernel height) / (stride + 1)

        #200x200
        #((200 - 5) / 2) + 1 = 98
        #((98 - 5) / 2) + 1 = 47
        #((47 - 3)) / 2) + 1 = 23
        #((23 - 2) / 1) + 1 = 22
        #((22 - 1) / 1) + 1 = 22


        self.fc1 = nn.Linear(64*34*34, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               
        x = self.relu(self.conv16(x))  
        x = self.relu(self.conv24(x))  
        x = self.relu(self.conv32(x))
        x = self.relu(self.conv48(x))  
        x = self.relu(self.conv64(x))    

        x = self.flatten(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc1(x)
        return x

# Dokney docs --> parts --> keras and fastai (uses PyTorch)
# donkeycar (github) --> donkeycar --> parts --> keras.py and fastai.py (try fastai first)
# https://github.com/autorope/donkeycar/blob/main/donkeycar/parts/fastai.py
class DonkeyNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv24 = nn.Conv2d(3, 24, kernel_size=(5, 5), stride=(2, 2))
        self.conv32 = nn.Conv2d(24, 32, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_5 = nn.Conv2d(32, 64, kernel_size=(5, 5), stride=(2, 2))
        self.conv64_3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

        self.fc1 = nn.Linear(64*30*30, 128)  # (64*30*30, 128) for 300x300 images
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):               #   300x300                     #  120x160 images
        x = self.relu(self.conv24(x))  # (300-5)/2+1 = 148     |     (120-5)/2+1 = 58   (160-5)/2+1 = 78
        x = self.relu(self.conv32(x))  # (148-5)/2+1 = 72      |     (58 -5)/2+1 = 27   (78 -5)/2+1 = 37
        x = self.relu(self.conv64_5(x))  # (72-5)/2+1 = 34     |     (27 -5)/2+1 = 12   (37 -5)/2+1 = 17
        x = self.relu(self.conv64_3(x))  # 34-3+1 = 32         |     12 - 3 + 1  = 10   17 - 3 + 1  = 15
        x = self.relu(self.conv64_3(x))  # 32-3+1 = 30         |     10 - 3 + 1  = 8    15 - 3 + 1  = 13

        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


