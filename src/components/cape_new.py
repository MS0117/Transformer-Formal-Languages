#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
from torch import Tensor
import math
from typing import Optional, Union

class CAPE1d(nn.Module):
  def __init__(self, d_model, max_global_shift, max_local_shift, max_global_scaling, normalize=False ,freq_scale=1.0):

    super().__init__()

    self.max_global_shift=max_global_shift
    self.max_local_shift=max_local_shift
    self.max_global_scaling=max_global_scaling
    self.normalize=normalize
    self.freq_scale=freq_scale

    freq=freq_scale*torch.exp(-2.0* torch.floor(torch.arange(d_model)/2)*(math.log(1e4)/d_model))
    self.register_buffer('freq', freq)

    cos_shift=(torch.pi/2.0)*(torch.arange(d_model)%2)
    self.register_buffer('cos_shift', cos_shift)

    self.register_buffer('content_scale', Tensor([math.sqrt(d_model)]))


  def forward(self,x, x_length: Optional[Tensor] = None,
                        position_delta: Optional[Union[int, Tensor]] = None):
                                          #ready for relative encoding

    batch_size=x.size(0)
    seq_len=x.size(1)
    position_vector=torch.arange(seq_len).repeat(batch_size,1).to(x)
        
    if position_delta is None:
            position_delta = 1
            
    if x_length is not None:
            x_length=x_length.unsqueeze(1).view([-1,1]).contiguous()        
            padding_mask=position_vector>x_length
            position_vector[padding_mask]=float('nan')

    if self.normalize:
      position_vector=position_vector-torch.nonmean(position_vector, dim=1,keepdim=True)


    position_vector=self.positional_embedding(position_vector, position_delta) 
    
    position_vector=position_vector.unsqueeze(-1)                                   #rearrange(positions, 'b t -> b t 1')로도 쓸 수 있음

    position_vector=position_vector*self.freq.to(x)                               #여기서 position_vector도 크기가 커짐...d_model 차원된다.

    positional_embedding=torch.sin(position_vector+self.cos_shift.to(x))

    positional_embedding=torch.nan_to_num(positional_embedding,nan=0).to(x)

    return (x*self.content_scale)+positional_embedding


  def positional_embedding(self,position_vector, positions_delta: Optional[Union[int, Tensor]] = None): 
      
    if self.training:
      batch_size=position_vector.size(0)
      seq_len=position_vector.size(1)

      global_shift=torch.FloatTensor(batch_size,1).uniform_(-self.max_global_shift,self.max_global_shift).to(position_vector.device)

      local_shift=torch.FloatTensor(batch_size,seq_len).uniform_(-self.max_local_shift,self.max_local_shift).to(position_vector.device)

      global_scaling=torch.FloatTensor(batch_size,1).uniform_(-self.max_global_scaling,self.max_global_scaling).to(position_vector.device)


      position_vector=(position_vector+global_shift+local_shift)*torch.exp(global_scaling)

    return position_vector

  def set_content_scale(self, content_scale: float):
     self.content_scale = Tensor([content_scale])