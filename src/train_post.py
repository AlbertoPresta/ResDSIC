
import torch
from compress.models import  PostProcessedNetwork, ChannelProgresssiveWACNN

if __name__ == "__main__":

    c = torch.randn(1,3,256,256)

    base_net = ChannelProgresssiveWACNN()
    post_net = PostProcessedNetwork(base_net)


    d = post_net(c) 
    print("lo shape di d è: ",d) 
