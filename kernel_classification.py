
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from funcEnv import funcEnv

points = 100
batch_size = 50
fc1_unit = 1500
conv_out = 48
class Data:
    def __init__(self,classes,seed = 0):
        self.funcEnv = funcEnv()
        self.classes = np.array(classes)
        self.rs = np.random.RandomState(seed)
    def genData(self,num_data):
        inputs = []
        labels = []
        lss = []
        for i in range(num_data):
            ls = self.rs.uniform(0.3,0.05)
            self.funcEnv.kernel_lengthscale = ls
            self.funcEnv.funType = self.rs.choice(self.classes)
            self.funcEnv.reset(sample_point=points)
            X = np.linspace(0,1,points)
            Y = self.funcEnv.curFun(X)
            label = np.zeros(self.classes.shape)
            label[np.where(self.classes == self.funcEnv.funType)] = 1
            # inputs.append(np.concatenate((X,Y)))
            inputs.append(Y)
            labels.append(label)
            lss.append(ls)

        inputs = np.array(inputs)
        labels = np.array(labels)
        lss = np.array(lss)
        return torch.from_numpy(inputs).float(),torch.from_numpy(labels).long(),torch.from_numpy(lss).float()

class classification_Net(nn.Module):
    def __init__(self,classes,use_conv = False):
        super(classification_Net, self).__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
            self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)
            # self.bn1 = nn.BatchNorm1d(1)
            self.fc1 = nn.Linear(conv_out, fc1_unit)
        else:
            self.fc1 = nn.Linear(points, 20)
        self.fc2 = nn.Linear(fc1_unit, len(classes))

    def forward(self, x):
        if self.use_conv:
            # (batch,channel,data)
            x = torch.reshape(x,(-1,1,points))
            x = self.conv1(x)
            x = self.conv2(x)
            # x = F.relu(self.bn1(x))
            # print(x.shape)
            x = torch.reshape(x,(-1,conv_out))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

class lengthscale_Net(nn.Module):
    def __init__(self,use_conv = False):
        super(lengthscale_Net, self).__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=2)
            self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1)
            # self.bn1 = nn.BatchNorm1d(1)
            self.fc1 = nn.Linear(conv_out, fc1_unit)
        else:
            self.fc1 = nn.Linear(points, 20)
        self.fc2 = nn.Linear(fc1_unit, 1)

    def forward(self, x):
        if self.use_conv:
            x = torch.reshape(x,(-1,1,points))
            x = self.conv1(x)
            x = self.conv2(x)
            # x = F.relu(self.bn1(x))
            # print(x.shape)
            x = torch.reshape(x,(-1,conv_out))
        x = F.relu(self.fc1(x))
        x = F.softplus(self.fc2(x))
        return x

class train_Net:
    def __init__(self,max_step=int(1e4)):
        classes = ["Exp","RQ","RBF","MA"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_step = max_step
        self.kernel_net = classification_Net(classes,use_conv=True).to(self.device)
        self.ls_net = lengthscale_Net(use_conv=True).to(self.device)
        self.ls_lr = 1e-4
        self.kernel_lr = 1e-3
        self.kernel_optimizer = optim.Adam(self.kernel_net.parameters(), lr=0.001)
        self.ls_optimizer = optim.Adam(self.ls_net.parameters(), lr=self.ls_lr)
        self.ls_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.ls_optimizer, 
                                                              T_max=self.max_step,
                                                              eta_min=self.ls_lr /100)
        self.kernel_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.kernel_optimizer, 
                                                              T_max=self.max_step,
                                                              eta_min=self.kernel_lr /100)
        self.data = Data(classes)
        self.MSE = nn.MSELoss().to(self.device)
        self.cross = nn.CrossEntropyLoss().to(self.device)

    def store_weight(self,step):
        with open("weight/kernel_weight_{}".format(step), "wb") as f:
            torch.save(self.kernel_net.state_dict(), f)
        with open("weight/ls_weight_{}".format(step), "wb") as f:
            torch.save(self.ls_net.state_dict(), f)

    def run(self):
        for i in range(self.max_step):

            inputs, labels, ls = self.data.genData(batch_size)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            ls = ls.to(self.device)
            # kernel
            self.kernel_optimizer.zero_grad()
            kernel_outputs = self.kernel_net(inputs)
            # print(kernel_outputs.shape,labels.shape)
            kernel_loss = self.cross(kernel_outputs, torch.max(labels, 1)[1])
            kernel_loss.backward()
            self.kernel_optimizer.step()
            self.kernel_scheduler.step(epoch=i)

            # length scale
            self.ls_optimizer.zero_grad()
            ls_output = self.ls_net(inputs)
            ls_loss = self.MSE(ls_output, ls)
            ls_loss.backward()
            self.ls_optimizer.step()
            self.ls_scheduler.step(epoch=i)

            if i % 1000 == 0 or i == self.max_step-1:
                self.store_weight(i)
                self.validation()
                print("kernel loss = {}, ls loss = {}, step = {}".format(kernel_loss,ls_loss,i))
    def validation(self):
        correct = 0
        total = 0
        error = 0
        with torch.no_grad():
            for i in range(1000):
                inputs, labels, ls = self.data.genData(1)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ls = ls.to(self.device)
                
                kernel_outputs = self.kernel_net(inputs)
                ls_output = self.ls_net(inputs)
                total += labels.size(0)
                correct += 1 if torch.argmax(kernel_outputs)==torch.argmax(labels) else 0
                error += torch.abs(ls_output - ls)
                # print(np.argmax(kernel_outputs).numpy()==np.argmax(labels).numpy())
        print('Accuracy of kernel, 1000 test: %.2f %%' % (100 * correct / total))
        print('mean error of ls, 1000 test: %.4f' % (error / total))



if __name__ == "__main__":
    t = train_Net()
    t.run()
