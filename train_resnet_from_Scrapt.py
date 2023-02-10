import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils
from tensorboardX import SummaryWriter
import os
import numpy as np
from datetime import datetime

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix




seed ="{:%d-%m-%Y_%H-%M-%S}".format(datetime.now())

batch_size = 32
n_classe = 3
img_size = 128
epochs = 100

save_dir = "../results"


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        
        #downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        #add identity
        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x


        
        
class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)
        
    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)        
        
def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    losses = []
    for batch_idx, (inputs, targets) in enumerate(train_loader): #Dataset e sua classes
        #print(inputs.shape)
        #print(targets.shape)
        targets = targets.type(torch.LongTensor)   # casting to long
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_cnn.zero_grad()
        outputs = model(inputs)
        #if isinstance(outputs, list):
            #outputs = outputs[0]

        loss = criterion_cnn(outputs, targets)# criterion
        losses.append(loss.item())
        loss.backward()
        optimizer_cnn.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #registrar no tensorborder acc e loss
        writer.add_scalar('Training/ACC_',100.*correct/total, (epoch*len(train_loader.dataset)/12)+batch_idx)
        writer.add_scalar('Training/loss_',train_loss/(batch_idx+1),(epoch*len(train_loader.dataset)/12)+batch_idx)
    #avg_loss = sum(losses)/len(losses)
    #scheduler.step(avg_loss)
    print('\n %d',correct/total*100)
    writer.add_scalar('Training/ACC',correct/total*100, epoch)


def val():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, list):
              outputs = outputs[0]
            _, predicted = outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print("ACC_test",acc)



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          titleG='Confusion matrix',
                          cmap=None,
                          normalize=True):
 
    import matplotlib.pyplot as plt_
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy
    print('Predição do  label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if cmap is None:
        cmap = plt_.get_cmap('Blues')

    plt_.figure()
    plt_.imshow(cm, interpolation='nearest', cmap=cmap)
    plt_.title(titleG)
    plt_.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt_.xticks(tick_marks, target_names, rotation=45)
        plt_.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt_.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt_.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt_.tight_layout()
    plt_.ylabel('Verdade do label')
    plt_.xlabel('Predicted label')
    plt_.xlabel('Predição do  label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt_.savefig(title+'-'+str(seed)+'resnet_confusion_matrix.png', dpi=120)
    #plt_.show() 
    plt_.close()
    

def convert_label_(pred_y,true_l,list_label):
  #true_l = np.array(true_l, int)
  y_true = []
  y_pred = []
  for i in range(len(pred_y)):
    for j in range(len(pred_y[i])):
      y_pred.append(list_label[pred_y[i][j].item()])
      y_true.append(list_label[np.array( true_l[i][j].item(), int)])
  return np.array(y_true),np.array(y_pred)


def MC_val():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    pred_ = []
    label_ = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, list):
              outputs = outputs[0]
            _, predicted = outputs.max(1)

            
            pred_.append(predicted.data.cpu().numpy())
            label_.append(targets.data.cpu().numpy())
            
        return pred_, label_


if __name__ == "__main__":

    # Use specific gpus
    #os.environ["CUDA_VISIBLE_DEVICES"] = 0
    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    train_images = np.load('../Data/_amostras_06012023_/train_amostras_06012023_images.npy') 
    train_labels_ = np.load('../Data/_amostras_06012023_/train_amostras_06012023_labels.npy')


    print("before=$%;  ",train_images.shape)
    #print(train_labels_c.size())

    t, w, h, c = train_images.shape
    dataTrain = train_images.reshape(t,c,w,h)

    print("After=$%;  ",dataTrain.shape)

    treino_data, teste_data, treino_label, teste_label = train_test_split(dataTrain, train_labels_, test_size=0.20, random_state=42)
        
    tensor_x_train = torch.Tensor(treino_data) 
    tensor_y_train = torch.Tensor(treino_label)

    tensor_x_test = torch.Tensor(teste_data) 
    tensor_y_test = torch.Tensor(teste_label)

    train_dataset = TensorDataset(tensor_x_train,tensor_y_train) # create your datset
    test_dataset = TensorDataset(tensor_x_test,tensor_y_test) # create your datset

    train_loader = DataLoader(train_dataset,batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=True, num_workers=0)
    
    model = ResNet152(n_classe).to(device)

    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        print("Using {} GPUs".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)
    
    #writer = SummaryWriter(save_dir + "/""runs/cnn_5bands_{:%d-%m-%Y_%H-%M-%S}".format(datetime.now()))
    writer = SummaryWriter("../runs/cnn_resnet152_{:%d-%m-%Y_%H-%M-%S}".format(datetime.now()))

    def save_model(model):
        torch.save(model.state_dict(), save_dir + "/""Seed_"+str(seed)+"_Resnet_model.pt")
    
    # Create loss criterion & optimizer
    criterion_cnn = nn.CrossEntropyLoss()
    #optimizer_cnn = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)
    
    optimizer_cnn = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_cnn, factor = 0.1, patience=5)

    for epoch in range(1, epochs):
        train(epoch)

    val()
     
    #Save model
    save_model(model)

    pred, label = MC_val()

    y_pred, y_true = convert_label_(pred,label,label_list)
    confusion = confusion_matrix(y_true, y_pred, labels = label_list)

    print(confusion)
    plot_confusion_matrix(cm = np.array(confusion),normalize = False, target_names = label_list, title =' Test', titleG = ' Test')
