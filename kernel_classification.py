
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from funcEnv import funcEnv

points = 100
class Data:
    def __init__(self,classes):
        self.funcEnv = funcEnv()
        self.classes = np.array(classes)
    def genData(self,num_data):
        inputs = []
        labels = []
        lss = []
        for i in range(num_data):
            ls = np.random.uniform(0.3,0.05)
            self.funcEnv.kernel_lengthscale = ls
            self.funcEnv.funType = np.random.choice(self.classes)
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
    def __init__(self,classes):
        super(classification_Net, self).__init__()
        self.fc1 = nn.Linear(points, 20)
        self.fc2 = nn.Linear(20, len(classes))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x

class lengthscale_Net(nn.Module):
    def __init__(self):
        super(lengthscale_Net, self).__init__()
        self.fc1 = nn.Linear(points, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softplus(self.fc2(x))
        return x

class train_Net:
    def __init__(self):
        classes = ["Exp","RQ","RBF","MA"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_step = int(1e4)
        self.kernel_net = classification_Net(classes).to(self.device)
        self.ls_net = lengthscale_Net().to(self.device)
        self.kernel_optimizer = optim.Adam(self.kernel_net.parameters(), lr=0.001)
        self.ls_optimizer = optim.Adam(self.ls_net.parameters(), lr=0.00001)
        self.data = Data(classes)
        self.MSE = nn.MSELoss().to(self.device)
        self.cross = nn.CrossEntropyLoss().to(self.device)

    def store_weight(self,step):
        with open("kernel_weight_{}".format(step), "wb") as f:
            torch.save(self.kernel_net.state_dict(), f)
        with open("ls_weight_{}".format(step), "wb") as f:
            torch.save(self.ls_net.state_dict(), f)

    def run(self):
        for i in range(self.max_step):

            inputs, labels, ls = self.data.genData(50)
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

            # length scale
            self.ls_optimizer.zero_grad()
            ls_output = self.ls_net(inputs)
            ls_loss = self.MSE(ls_output, ls)
            ls_loss.backward()
            self.ls_optimizer.step()

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
                correct += 1 if np.argmax(kernel_outputs.cpu()).numpy()==np.argmax(labels.cpu()).numpy() else 0
                error += np.abs(ls_output.cpu() - ls.cpu())
                # print(np.argmax(kernel_outputs).numpy()==np.argmax(labels).numpy())
        print('Accuracy of kernel, 1000 test: %d %%' % (100 * correct / total))
        print('mean error of ls, 1000 test: {}'.format(error.cpu().numpy() / total))



if __name__ == "__main__":
    t = train_Net()
    t.run()
