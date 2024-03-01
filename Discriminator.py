import torch
import torch.nn as nn

class cnnblock(nn.Module):
    def __init__(self,in_channel,out_channel,stride):
        super(cnnblock,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,4,stride,1,padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self,x):
       return  self.conv(x)



class discriminator(nn.Module):
    def __init__(self, inchannels=3, features=[64,128,256,512]):
        super(discriminator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=inchannels*2, out_channels=features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(cnnblock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
            
        self.model = nn.Sequential(*layers)
        self.final_conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        x = self.final_conv(x)
        return x


def test():
    x=torch.randn((1,3,256,256))
    y=torch.randn((1,3,256,256))

    model=discriminator(inchannels=3)
    pred=model(x,y)

    print(model)
    print(pred.shape)
    

if __name__=="__main__":
    test()


        
         
    
