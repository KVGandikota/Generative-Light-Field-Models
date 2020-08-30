"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from pytorch_conv4d.conv4d import Conv4d

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

    
class WAE_lf_7x7(nn.Module):
    def __init__(self, z_dim=180):
        super(WAE_lf, self).__init__()
        self.z_dim = z_dim
        self.encoder_s1x = nn.Sequential(
            nn.Conv3d(7,28,(3,3,3),(1,1,1),(0,0,0),bias=False,groups = 7),
            nn.BatchNorm3d(28),
            nn.ReLU(True),
            nn.Conv3d(28,49,(3,3,3),(1,2,2),(0,0,0), bias=False,groups = 7),
            nn.BatchNorm3d(49),
            nn.ReLU(True),
            nn.Conv3d(49,60,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(60),
            nn.ReLU(True)
        )
        self.encoder_s1y = nn.Sequential(
            nn.Conv3d(7,28,(3,3,3),(1,1,1),(0,0,0),bias=False,groups = 7),
            nn.BatchNorm3d(28),
            nn.ReLU(True),
            nn.Conv3d(28,49,(3,3,3),(1,2,2),(0,0,0), bias=False,groups = 7),
            nn.BatchNorm3d(49),
            nn.ReLU(True),
            nn.Conv3d(49,60,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(60),
            nn.ReLU(True)
        )
        self.encoder_s2 = nn.Sequential(
            nn.Conv3d(120+20,200,(1,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(200),
            nn.ReLU(True),
            nn.Conv3d(200,250,(1,3,3),(1,2,2),(0,0,0),bias=False),
            nn.BatchNorm3d(250),
            nn.ReLU(True),
            nn.Conv3d(250,300,(1,3,3),(1,1,1),(0,0,0),bias=False),          
            View((-1, 300))
        )        
        self.cv_e1 = nn.Sequential(
            nn.Conv2d(1,  6, (3,3), (1,1), (0,0), bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6, 10, (3,3), (2,2), (0,0), bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 20, (3,3), (1,1), (0,0), bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(True)
        )
        self.cv_e2 = nn.Sequential(
            nn.Conv2d(20, 40, (3,3), (1,1), (0,0), bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            nn.Conv2d(40, 50, (3,3), (2,2), (0,0), bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Conv2d(50, 60, (3,3), (1,1), (0,0), bias=False),
            View((-1, 60))
        )
        self.final_4d = nn.Sequential(
            Conv4d(8, 1, (3,3,3,3), 1,1, bias=False)            
        )
        self.final_2d = nn.Sequential(
            nn.Conv2d(50, 49, (1,1), (1,1), (0,0), bias=False)            
        )
        self.fcl1 =  nn.Sequential(
            nn.Linear(300+60, 300),
            nn.ReLU(True)
        )
        self.fcl2 =  nn.Linear(300,z_dim)
        self.dfc1 =  nn.Linear(z_dim+60,300)
        self.dfc2 =  nn.Linear(40,60)        
        
        self.decoder_s1 = nn.Sequential(
            View((-1, 300, 1, 1,1)),
            nn.ConvTranspose3d(300,250,(1,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(250),
            nn.ReLU(True),
            nn.ConvTranspose3d(250,200,(1,3,3),(1,2,2),(0,0,0),bias=False),
            nn.BatchNorm3d(200),
            nn.ReLU(True),
            nn.ConvTranspose3d(200,120,(1,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(120),
            nn.ReLU(True)
        )
        self.decoder_s2x = nn.Sequential(
            nn.ConvTranspose3d(140,84,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(84),
            nn.ReLU(True),
            nn.ConvTranspose3d(84,49,(3,3,3),(1,2,2),(0,0,0),bias=False,groups=7),
            nn.BatchNorm3d(49),
            nn.ReLU(True),
            nn.ConvTranspose3d(49,28,(3,3,3),(1,1,1),(0,0,0),bias=False,groups=7)
        )     
        self.decoder_s2y = nn.Sequential(
            nn.ConvTranspose3d(140,84,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(84),
            nn.ReLU(True),
            nn.ConvTranspose3d(84,49,(3,3,3),(1,2,2),(0,0,0),bias=False,groups=7),
            nn.BatchNorm3d(49),
            nn.ReLU(True),
            nn.ConvTranspose3d(49,28,(3,3,3),(1,1,1),(0,0,0),bias=False,groups=7)
        )  
    def _encode(self, x):
        cv = x[:,2,2,:,:]
        cv = torch.reshape(cv,(cv.shape[0],1,25,25))
        cvf1 = self.cv_e1(cv)
        cvf2 = self.cv_e2(cvf1)
        cvf1 = cvf1.reshape(cv.shape[0],20,1,9,9)        
        codedx1 = self.encoder_s1x(x)
        codedy1 = self.encoder_s1y(x.permute(0,2,1,3,4))
        coded1 = torch.cat((codedx1,codedy1,cvf1),1)
        code = self.encoder_s2(coded1)
        xat = torch.cat((code,cvf2),1)
        coded = self.fcl1(xat)
        coded = self.fcl2(coded)
        return coded
    def _decode(self, z,cv):
        # downsample input image (central view)
        cvf1 = self.cv_e1(cv)
        cvf2 = self.cv_e2(cvf1)
        cvf1 = cvf1.reshape(cv.shape[0],20,1,9,9)        
        z = torch.cat((z,cvf2),1)
        z = self.dfc1(z)
        dec1 = self.decoder_s1(z)
        dec1 = torch.cat((dec1,cvf1),1)
        decx = self.decoder_s2x(dec1)
        decy = self.decoder_s2y(dec1)                        
        decy = decy.permute(0,2,1,3,4)        
        decx = decx.reshape(cv.shape[0],4,7,7,25,25)
        decy = decx.reshape(cv.shape[0],4,7,7,25,25)
        dec = torch.cat((decx,decy),1)
        dec = self.final_4d(dec)
        dec = dec.reshape(cv.shape[0],49,25,25)
        dec = torch.cat((dec,cv),1)
        dec = self.final_2d(dec)
        dec = dec.reshape(cv.shape[0],7,7,25,25)
        return dec
    def forward(self, x):
        z = self._encode(x)        
        cv = x[:,3,3,:,:]
        cv = torch.reshape(cv,(cv.shape[0],1,25,25))
        x_recon = self._decode(z,cv)
        return x_recon, z    
        
class WAE_lf_5x5(nn.Module):
    def __init__(self, z_dim=180):
        super(WAE_lf, self).__init__()
        self.z_dim = z_dim
        self.encoder_s1x = nn.Sequential(
            nn.Conv3d(5,20,(3,3,3),(1,1,1),(0,0,0),bias=False,groups = 5),
            nn.BatchNorm3d(20),
            nn.ReLU(True),
            nn.Conv3d(20,40,(3,3,3),(1,2,2),(1,0,0), bias=False,groups = 5),
            nn.BatchNorm3d(40),
            nn.ReLU(True),
            nn.Conv3d(40,60,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(60),
            nn.ReLU(True)
        )
        self.encoder_s1y = nn.Sequential(
            nn.Conv3d(5,20,(3,3,3),(1,1,1),(0,0,0),bias=False,groups = 5),
            nn.BatchNorm3d(20),
            nn.ReLU(True),
            nn.Conv3d(20,40,(3,3,3),(1,2,2),(1,0,0), bias=False,groups = 5),
            nn.BatchNorm3d(40),
            nn.ReLU(True),
            nn.Conv3d(40,60,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(60),
            nn.ReLU(True)
        )
        self.encoder_s2 = nn.Sequential(
            nn.Conv3d(120+20,200,(1,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(200),
            nn.ReLU(True),
            nn.Conv3d(200,250,(1,3,3),(1,2,2),(0,0,0),bias=False),
            nn.BatchNorm3d(250),
            nn.ReLU(True),
            nn.Conv3d(250,300,(1,3,3),(1,1,1),(0,0,0),bias=False),          
            View((-1, 300))
        )        
        self.cv_e1 = nn.Sequential(
            nn.Conv2d(1,  6, (3,3), (1,1), (0,0), bias=False),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6, 10, (3,3), (2,2), (0,0), bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(True),
            nn.Conv2d(10, 20, (3,3), (1,1), (0,0), bias=False),
            nn.BatchNorm2d(20),
            nn.ReLU(True)
        )
        self.cv_e2 = nn.Sequential(
            nn.Conv2d(20, 40, (3,3), (1,1), (0,0), bias=False),
            nn.BatchNorm2d(40),
            nn.ReLU(True),
            nn.Conv2d(40, 50, (3,3), (2,2), (0,0), bias=False),
            nn.BatchNorm2d(50),
            nn.ReLU(True),
            nn.Conv2d(50, 60, (3,3), (1,1), (0,0), bias=False),
            View((-1, 60))
        )
        self.final_4d = nn.Sequential(
            Conv4d(8, 1, (3,3,3,3), 1,1, bias=False)            
        )
        self.final_2d = nn.Sequential(
            nn.Conv2d(26, 25, (1,1), (1,1), (0,0), bias=False)            
        )
        self.fcl1 =  nn.Sequential(
            nn.Linear(300+60, 300),
            nn.ReLU(True)
        )
        self.fcl2 =  nn.Linear(300,z_dim)
        self.dfc1 =  nn.Linear(z_dim+60,300)
        self.dfc2 =  nn.Linear(40,60)        
        
        self.decoder_s1 = nn.Sequential(
            View((-1, 300, 1, 1,1)),
            nn.ConvTranspose3d(300,250,(1,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(250),
            nn.ReLU(True),
            nn.ConvTranspose3d(250,200,(1,3,3),(1,2,2),(0,0,0),bias=False),
            nn.BatchNorm3d(200),
            nn.ReLU(True),
            nn.ConvTranspose3d(200,120,(1,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(120),
            nn.ReLU(True)
        )
        self.decoder_s2x = nn.Sequential(
            nn.ConvTranspose3d(140,80,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(80),
            nn.ReLU(True),
            nn.ConvTranspose3d(80,40,(3,3,3),(1,2,2),(1,0,0),bias=False,groups=5),
            nn.BatchNorm3d(40),
            nn.ReLU(True),
            nn.ConvTranspose3d(40,20,(3,3,3),(1,1,1),(0,0,0),bias=False,groups=5)
        )     
        self.decoder_s2y = nn.Sequential(
            nn.ConvTranspose3d(140,80,(3,3,3),(1,1,1),(0,0,0),bias=False),
            nn.BatchNorm3d(80),
            nn.ReLU(True),
            nn.ConvTranspose3d(80,40,(3,3,3),(1,2,2),(1,0,0),bias=False,groups=5),
            nn.BatchNorm3d(40),
            nn.ReLU(True),
            nn.ConvTranspose3d(40,20,(3,3,3),(1,1,1),(0,0,0),bias=False,groups=5)
        )  
    def _encode(self, x):
        cv = x[:,2,2,:,:]
        cv = torch.reshape(cv,(cv.shape[0],1,25,25))
        cvf1 = self.cv_e1(cv)
        cvf2 = self.cv_e2(cvf1)
        cvf1 = cvf1.reshape(cv.shape[0],20,1,9,9)        
        codedx1 = self.encoder_s1x(x)
        codedy1 = self.encoder_s1y(x.permute(0,2,1,3,4))
        coded1 = torch.cat((codedx1,codedy1,cvf1),1)
        code = self.encoder_s2(coded1)
        xat = torch.cat((code,cvf2),1)
        coded = self.fcl1(xat)
        coded = self.fcl2(coded)
        return coded
    def _decode(self, z,cv):
        # downsample input image (central view)
        cvf1 = self.cv_e1(cv)
        cvf2 = self.cv_e2(cvf1)
        cvf1 = cvf1.reshape(cv.shape[0],20,1,9,9)        
        z = torch.cat((z,cvf2),1)
        z = self.dfc1(z)
        dec1 = self.decoder_s1(z)
        dec1 = torch.cat((dec1,cvf1),1)
        decx = self.decoder_s2x(dec1)
        decy = self.decoder_s2y(dec1)                        
        decy = decy.permute(0,2,1,3,4)        
        decx = decx.reshape(cv.shape[0],4,5,5,25,25)
        decy = decx.reshape(cv.shape[0],4,5,5,25,25)
        dec = torch.cat((decx,decy),1)
        dec = self.final_4d(dec)
        dec = dec.reshape(cv.shape[0],25,25,25)
        dec = torch.cat((dec,cv),1)
        dec = self.final_2d(dec)
        dec = dec.reshape(cv.shape[0],5,5,25,25)
        return dec
    def forward(self, x):
        z = self._encode(x)        
        cv = x[:,2,2,:,:]
        cv = torch.reshape(cv,(cv.shape[0],1,25,25))
        x_recon = self._decode(z,cv)
        return x_recon, z    
