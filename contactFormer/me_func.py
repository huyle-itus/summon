import torch
import MinkowskiEngine as ME

def create_me_tensor(feats):
    c = feats.shape[0]
    coords = torch.tensor([[i,0] for i in range(c)]).cuda()
    return ME.SparseTensor(coordinates=coords,
                           features=feats)