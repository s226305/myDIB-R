import torch
import torch.nn
import torch.autograd
from torch.autograd import Function

import cv2
import numpy as np

#build.pyを実行すると作成される
import dr_batch_dib_render as dr_cuda_batch

from rendersettings import height, width, \
expand, knum, \
multiplier, eps, debug

############################################
delta = 7000  # int(1.0 / (3e-3))

############################################
# Inherit from Function
class TriRender2D(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, tfpoints3d_bxfx9, tfpoints2d_bxfx6, \
                tfnormalz_bxfx1, tffeatures_bxfx3d):
        
        bnum = tfpoints3d_bxfx9.shape[0]
        fnum = tfpoints3d_bxfx9.shape[1]

        
        #数値エラーを回避する
        tfpoints2dmul_bxfx6 = multiplier * tfpoints2d_bxfx6
        
        # bbox
        tfpoints2d_bxfx3x2 = tfpoints2dmul_bxfx6.view(bnum, fnum, 3, 2)
        tfpoints_min = torch.min(tfpoints2d_bxfx3x2, dim=2)[0]
        tfpoints_max = torch.max(tfpoints2d_bxfx3x2, dim=2)[0]
        tfpointsbbox_bxfx4 = torch.cat((tfpoints_min, tfpoints_max), dim=2)
        
        # bbox2
        tfpoints_min = tfpoints_min - expand * multiplier;
        tfpoints_max = tfpoints_max + expand * multiplier;
        tfpointsbbox2_bxfx4 = torch.cat((tfpoints_min, tfpoints_max), dim=2)
        
        # depth
        tfpointsdep_bxfx1 = (tfpoints3d_bxfx9[:, :, 2:3] \
        +tfpoints3d_bxfx9[:, :, 5:6] \
        +tfpoints3d_bxfx9[:, :, 8:9]) / 3.0
        
        tfimidxs_bxhxwx1 = torch.zeros(bnum, height, width, 1, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfimdeps_bxhxwx1 = -1000 * torch.ones(bnum, height, width, 1, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfimweis_bxhxwx3 = torch.zeros(bnum, height, width, 3, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfims_bxhxwxd = torch.zeros(bnum, height, width, 3, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfimprob_bxhxwx1 = torch.zeros(bnum, height, width, 1, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        
        #  中間変数(?)
        tfprobface = torch.zeros(bnum, height, width, knum, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfprobcase = torch.zeros(bnum, height, width, knum, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfprobdis = torch.zeros(bnum, height, width, knum, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfprobdep = torch.zeros(bnum, height, width, knum, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        tfprobacc = torch.zeros(bnum, height, width, knum, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
         
        tfpointsdirect_bxfx1 = tfnormalz_bxfx1.contiguous()
        dr_cuda_batch.forward(tfpoints3d_bxfx9, tfpoints2dmul_bxfx6, \
                              tfpointsdirect_bxfx1, tfpointsbbox_bxfx4, \
                              tfpointsbbox2_bxfx4, tfpointsdep_bxfx1, \
                              tffeatures_bxfx3d,
                              tfimidxs_bxhxwx1, tfimdeps_bxhxwx1, \
                              tfimweis_bxhxwx3, \
                              tfprobface, tfprobcase, tfprobdis, tfprobdep, tfprobacc, \
                              tfims_bxhxwxd, tfimprob_bxhxwx1,
                              multiplier, delta)
        
        debug_im = torch.zeros(bnum, height, width, 3, dtype=torch.float32).to(tfpoints2d_bxfx3x2.device)
        ctx.save_for_backward(tfims_bxhxwxd, tfimprob_bxhxwx1, \
                              tfimidxs_bxhxwx1, tfimweis_bxhxwx3, \
                              tfpoints2dmul_bxfx6, tffeatures_bxfx3d, \
                              tfprobface, tfprobcase, tfprobdis, tfprobdep, tfprobacc,
                              debug_im)   
        
        tfims_bxhxwxd.requires_grad = True
        tfimprob_bxhxwx1.requires_grad = True
        return tfims_bxhxwxd, tfimprob_bxhxwx1
        
    # This function has only a single output, so it gets only one gradient
    @staticmethod 
    def backward(ctx, dldI_bxhxwxd, dldp_bxhxwx1):
        
        tfims_bxhxwxd, tfimprob_bxhxwx1, \
        tfimidxs_bxhxwx1, tfimweis_bxhxwx3, \
        tfpoints2dmul_bxfx6, tfcolors_bxfx3d, \
        tfprobface, tfprobcase, tfprobdis, tfprobdep, tfprobacc, \
        debug_im = ctx.saved_tensors

        dldp2 = torch.zeros_like(tfpoints2dmul_bxfx6)
        dldp2_prob = torch.zeros_like(tfpoints2dmul_bxfx6)
        dldc = torch.zeros_like(tfcolors_bxfx3d)

        dr_cuda_batch.backward(dldI_bxhxwxd.contiguous(), \
                               dldp_bxhxwx1.contiguous(), \
                               tfims_bxhxwxd, tfimprob_bxhxwx1, \
                               tfimidxs_bxhxwx1, tfimweis_bxhxwx3, \
                               tfprobface, tfprobcase, tfprobdis, tfprobdep, tfprobacc, \
                               tfpoints2dmul_bxfx6, tfcolors_bxfx3d, \
                               dldp2, dldc, dldp2_prob, \
                               debug_im, multiplier, delta)

        return None, dldp2 + dldp2_prob, None, dldc


###############################################################
linear = TriRender2D.apply

######################################################