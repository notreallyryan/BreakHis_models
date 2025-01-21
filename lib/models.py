from torchvision import models
from torch import nn

class VGG_model(nn.Module):
    """
    VGG16 based model with new classification layers for classifying Breast cancer histopathological data
    """
    def __init__(self):
        super(VGG_model, self).__init__()
        
        #getting transfer learning model
        self.vgg16 = models.vgg16(pretrained=True)
        
        #freezing layers
        for param in self.vgg16.parameters(): param.requires_grad = False

        #changing final layer to identity layer
        self.vgg16.classifier = nn.Identity()
        
        #changing classifier
        self.custom_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(25088,512),
            nn.Sigmoid(),
            nn.Linear(512,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(75,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = self.custom_classifier(x)
        return x

class ResNet_model(nn.Module):
    """
    ResNet based model with new classification layers for classifying Breast cancer histopathological data
    """
    def __init__(self):
        super(ResNet_model, self).__init__()
        #getting model
        self.resnet50 = models.resnet50(pretrained = True)
        
        #freezing weights
        for param in self.resnet50.parameters(): param.requires_grad = False

        #changing final layer to identity layer
        self.resnet50.fc = nn.Identity()
    
        #redefining classifier
        self.custom_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048,512),
            nn.Sigmoid(),
            nn.Linear(512,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(75,2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.resnet50(x)
        x = self.custom_classifier(x)
        return x

class MobileNet_model(nn.Module):
    """
    MobileNet based model with new classification layers for classifying Breast cancer histopathological data
    """
    def __init__(self):
        super(MobileNet_model, self).__init__()
        
        self.mobilenet_v2 = models.mobilenet_v2(pretrained=True)
    
        for param in self.mobilenet_v2.parameters(): param.requires_grad = False

        self.mobilenet_v2.classifier = nn.Identity()
       
        self.custom_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280,512),
            nn.Sigmoid(),
            nn.Linear(512,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(75,2),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.mobilenet_v2(x)
        x = self.custom_classifier(x)
        return x

class DenseNet_model(nn.Module):
    """
    DenseNet based model with new classification layers for classifying Breast cancer histopathological data
    """

    def __init__(self):
        super(DenseNet_model, self).__init__()
        self.dense_model = models.densenet201(pretrained = True)
    
        for param in self.dense_model.parameters(): param.requires_grad = False

        self.dense_model.classifier = nn.Identity()
                
        self.custom_classifier = nn.Sequential(
            nn.Linear(1920,512),
            nn.Sigmoid(),
            nn.Linear(512,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Linear(75,75),
            nn.Sigmoid(),
            nn.Dropout(0.3),
            nn.Linear(75,2),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.dense_model(x)
        x = self.custom_classifier(x)
        return x
