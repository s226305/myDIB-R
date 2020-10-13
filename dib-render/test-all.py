from __future__ import print_function
from __future__ import division

import pdb 

import torch
import torchvision.utils as vutils

import os
import numpy as np
from config import get_args

import sys
sys.path.append('../utils/')
sys.path.append('./render_cuda')

from dataloader import get_data_loaders
from mesh import loadobj, face2edge, edge2face, face2pneimtx, mtx2tfsparse, savemesh, savemeshcolor
from AutoEncoder import Ecoder
from perspective import camera_info_batch, perspectiveprojectionnp
from renderfunc_cluster import rendermeshcolor as rendermesh
from render_cuda.utils_render_color2 import linear

############################################
# Make experiments reproducible
torch.manual_seed(123456)
np.random.seed(123456)
eps = 1e-15

############################################

args = get_args()
args.img_dim = 64
args.batch_size = 64

FILELIST = args.filelist
IMG_DIM = args.img_dim
N_CHANNELS = args.img_channels
BATCH_SIZE = args.batch_size
TOTAL_EPOCH = args.epoch
ITERS_PER_LOG = args.iter_log
ITERS_PER_SAMPLE = args.iter_sample
ITERS_PER_MODEL = args.iter_model
VERBOSE = True

print('------------------')
print('| Configurations |')
print('------------------')
print('')
print('IMG_DIM:         {}'.format(IMG_DIM))
print('N_CHANNELS:      {}'.format(N_CHANNELS))
print('BATCH_SIZE:      {}'.format(BATCH_SIZE))
print('FILELIST:        {}'.format(FILELIST))
print('TOTAL_EPOCH:     {}'.format(TOTAL_EPOCH))
print('ITERS_PER_LOG:   {}'.format(ITERS_PER_LOG))
print('VERBOSE:         {}'.format(VERBOSE))
print('')

##########################################################
lossname = args.loss
cameramode = args.camera
viewnum = args.view

test_iter_num = args.iter
svfolder = args.svfolder
g_model_dir = args.g_model_dir
data_folder = args.data_folder

####################
# Load the dataset #
####################

#テスト用の場合視点は1つ
viewnum = 1

filelist = FILELIST
imsz = IMG_DIM
numworkers = args.thread
data = get_data_loaders(filelist, imsz, viewnum, mode='test', bs=BATCH_SIZE, numworkers=numworkers,data_folder=data_folder)

#pdb.set_trace()

############################################
# template球をロード
# sphere.obj: 642の頂点と1280の面がある
pointnp_px3, facenp_fx3 = loadobj('sphere.obj')
edge_ex2 = face2edge(facenp_fx3)
edgef_ex2 = edge2face(facenp_fx3, edge_ex2)
pneimtx = face2pneimtx(facenp_fx3)

pnum = pointnp_px3.shape[0]    #頂点数
fnum = facenp_fx3.shape[0]     #面数
enum = edge_ex2.shape[0]       #辺数

camfovy = 49.13434207744484 / 180.0 * np.pi
camprojmtx = perspectiveprojectionnp(camfovy, 1.0)    #質問したい質問したい質問したい質問したい質問したい質問したい質問したい質問したい

#pdb.set_trace()

################################################
# Define device, neural nets, optimizers, etc. #
################################################

# Automatic GPU/CPU の設置
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# モデル生成
g_model_im2mesh = Ecoder(N_CHANNELS, N_KERNELS=5, BATCH_SIZE=BATCH_SIZE, IMG_DIM=IMG_DIM, VERBOSE=VERBOSE).to(device)

#pdb.set_trace() 
        
############
# Training #
############

def test(iter_num=-1):
    print('Begin Testing!')
    #iter_num(おそらく学習回数)に基づいて最新の既存のチェックポイントをロード(g_model_dir:pretrained_modelの詳細が不明)
    #g_model_im2mesh.load_state_dict(torch.load(g_model_dir), strict=True)    #load_state_dict：学習済みパラメータの取得
    #g_model_im2mesh.eval()                                                   #eval：中身を文字列として実行
    #print('Loaded the latest checkpoints from {}'.format(g_model_dir))
    
    #テストサンプル用のディレクトリ
    model_iter = iter_num
    if not os.path.exists(os.path.join(svfolder, 'test-%d' % model_iter)):  #os.path.exists；ディレクトリ(ファイル)の存在確認
        print('Make Save Dir')                                               #os.path.join：文字列を結合し，パスにできる
        os.makedirs(os.path.join(svfolder, 'test-%d' % model_iter))          #svfolder(experiment)とtest--1を結合してパス名にし，ディレクトリを作成
    
    #グローバル変数(関数の外で宣言した変数)に値を代入したい場合はglobal宣言がいる
    global pointnp_px3, facenp_fx3, edgef_ex2, pneimtx
    global camprojmtx
    global cameramode, lossname
    
    #頂点数の半分の数をpに代入
    p = pointnp_px3
    pmax = np.max(p, axis=0, keepdims=True)
    pmin = np.min(p, axis=0, keepdims=True)
    pmiddle = (pmax + pmin) / 2
    p = p - pmiddle
    
    #よくわからん
    assert cameramode == 'per', 'now we only support perspective'
    pointnp_px3 = p * 0.35

    #numpyからpytorchへの変換
    tfp_1xpx3 = torch.from_numpy(pointnp_px3).to(device).view(1, pnum, 3)    #頂点数
    tff_fx3 = torch.from_numpy(facenp_fx3).to(device)                        #面数
    tfcamproj = torch.from_numpy(camprojmtx).to(device)                      #おそらくカメラアングル

    #初期化
    iou = {}        #iou評価指標
    catenum = {}    #おそらくカテゴリー数
    cates = '02691156,02828884,02933112,02958343,03001627,03211117,03636649,03691459,04090263,04256520,04379243,04401088,04530566'
    cates = cates.split(',')
    for ca in cates:
        iou[ca] = 0
        catenum[ca] = 0

    #学習回数を0で初期化し，学習スタート
    iter_num = 0
    for i, da in enumerate(data):    #enumrate:iには要素番号，daには要素の中身が順に格納される
        if da is None:
            continue
        
        iter_num += 1
        tfims = []
        tfcams = []

        #viewnumが1なのでループは1回
        #まだよくわかってない
        for j in range(viewnum):
            imnp = da['view%d' % j]['im']
            bs = imnp.shape[0]
            imnp_bxcxhxw = np.transpose(imnp, [0, 3, 1, 2])
            tfim_bx4xhxw = torch.from_numpy(imnp_bxcxhxw).to(device)
            tfims.append(tfim_bx4xhxw)
            # camera
            camrot_bx3x3 = da['view%d' % j]['camrot']
            campos_bx3 = da['view%d' % j]['campos']
            tfcamrot = torch.from_numpy(camrot_bx3x3).to(device)
            tfcampos = torch.from_numpy(campos_bx3).to(device)
            tfcameras = [tfcamrot, tfcampos, tfcamproj]
            tfcams.append(tfcameras)
            
        ########################################3
        with torch.no_grad():
            meshes = []
            meshcolors = []
            meshmovs = []
            
            #わかってないわかってないわかってないわかってない
            #j個のメッシュを作成
            for j in range(viewnum):
                meshmov_bxp3, mc_bxp3 = g_model_im2mesh(tfims[j][:, :args.img_channels,:,:])
                meshmov_bxpx3 = meshmov_bxp3.view(bs, -1, 3)    #view:配列のサイズ変更に用いる(-1がある次元は自動で変形してくれる)
                mesh_bxpx3 = meshmov_bxpx3 + tfp_1xpx3
                mc_bxpx3 = mc_bxp3.view(bs, -1, 3)

                # normalize
                mesh_max = torch.max(mesh_bxpx3, dim=1, keepdim=True)[0]    #keepdim:入力と出力の配列のサイズを同じにするか否か
                mesh_min = torch.min(mesh_bxpx3, dim=1, keepdim=True)[0]    #おそらくdim=1は1列目という意味
                mesh_middle = (mesh_min + mesh_max) / 2
                mesh_bxpx3 = mesh_bxpx3 - mesh_middle

                bs = mesh_bxpx3.shape[0]
                mesh_biggest = torch.max(mesh_bxpx3.view(bs, -1), dim=1)[0]
                mesh_bxpx3 = mesh_bxpx3 / mesh_biggest.view(bs, 1, 1) * 0.45

                meshes.append(mesh_bxpx3)
                meshcolors.append(mc_bxpx3)
                meshmovs.append(meshmov_bxpx3)
            
            meshesvv = []
            mcvv = []
            tfcamsvv = [[], [], tfcamproj]
            gtvv = []

            #わかってないわかってないわかってないわかってない
            # use j-th mesh
            for j in range(viewnum):
                # generate with k-th camera
                for k in range(viewnum):
                    mesh_bxpx3 = meshes[j]      #メッシュ
                    mc_bxpx3 = meshcolors[j]    #色
                    meshesvv.append(mesh_bxpx3)
                    mcvv.append(mc_bxpx3)
                    tfcamrot_bx3x3, tfcampos_bx3, _ = tfcams[k]
                    tfcamsvv[0].append(tfcamrot_bx3x3)
                    tfcamsvv[1].append(tfcampos_bx3)
                    # k-th camera, k-th image
                    tfim_bx4xhxw = tfims[k]
                    gtvv.append(tfim_bx4xhxw)
            
            #わかってないわかってないわかってないわかってない     
            mesh_vvbxpx3 = torch.cat(meshesvv)    #cat:結合
            mc_vvbxpx3 = torch.cat(mcvv)
            tfcamsvv[0] = torch.cat(tfcamsvv[0])
            tfcamsvv[1] = torch.cat(tfcamsvv[1])
            tmp, _ = rendermesh(mesh_vvbxpx3, mc_vvbxpx3, tff_fx3, tfcamsvv, linear)
            impre_vvbxhxwx3, silpred_vvbxhxwx1 = tmp
            
            ########################################
            
            #ここからここからここからここからここから
            #損失計算 
            
            #colorの損失計算(L1ノルム)
            tfim_vvbx4xhxw = torch.cat(gtvv)

            impre_vvbx3xhxw = impre_vvbxhxwx3.permute(0, 3, 1, 2)
            imgt_vvbx3xhxw = tfim_vvbx4xhxw[:, :3, :, :]
            colloss = 3 * torch.mean(torch.abs(impre_vvbx3xhxw - imgt_vvbx3xhxw))    #meanは平均をとっている
            
            #iouの損失計算
            #silpred_vvbx1xhxwが本来のシルエット
            #silgtが予測値
            silpred_vvbx1xhxw = silpred_vvbxhxwx1.view(viewnum * viewnum * bs, 1, IMG_DIM, IMG_DIM)
            silgt = tfim_vvbx4xhxw[:, 3:4, :, :]

            silmul = silpred_vvbx1xhxw * silgt
            siladd = silpred_vvbx1xhxw + silgt
            silmul = silmul.view(bs, -1)
            siladd = siladd.view(bs, -1)
            iouup = torch.sum(silmul, dim=1)
            ioudown = torch.sum(siladd - silmul, dim=1)
            iouneg = iouup / (ioudown + eps)
            silloss = 1.0 - torch.mean(iouneg)
            
        iouneg = iouneg.detach().cpu().numpy()
        for cid, ca in enumerate(da['cate']):    #enumrate:インデックスと中身を同時に取り出す
            iou[ca] += iouneg[cid]               #上で計算した損失を格納している
            catenum[ca] += 1

        if iter_num % 100 == 0:
            # 統計を出力する
            print('epo: {}, iter: {}, color_loss: {}, iou_loss: {}'. \
                  format(0, iter_num, colloss, silloss))
            
        #iter_num() / ITERS_PER_SAMPLE(defaultでは1000) の余りが999のとき真
        if iter_num % ITERS_PER_SAMPLE == ITERS_PER_SAMPLE - 1:
            silpred_vvbx3xhxw = silpred_vvbx1xhxw.repeat(1, 3, 1, 1)
            silgt_vvbx3xhxw = tfim_vvbx4xhxw[:, 3:4, :, :].repeat(1, 3, 1, 1)
            re = torch.cat((imgt_vvbx3xhxw, silgt_vvbx3xhxw, impre_vvbx3xhxw, silpred_vvbx3xhxw), dim=3)
            real_samples_dir = os.path.join(svfolder, 'test-%d' % model_iter, 'real_{:0>7d}.png'.format(iter_num))
            vutils.save_image(re, real_samples_dir, normalize=False)    #save_image:pytorchの関数で画像の保存を自動で行ってくれる
        
        #わかってないわかってないわかってないわかってない
        meshnp_bxpx3 = mesh_vvbxpx3.detach().cpu().numpy()
        meshcolnp_bxpx3 = mc_vvbxpx3.detach().cpu().numpy()
        meshcolnp_bxpx3[meshcolnp_bxpx3 < 0] = 0
        meshcolnp_bxpx3[meshcolnp_bxpx3 > 1] = 1
        meshcolnp_bxpx3 = meshcolnp_bxpx3[..., ::-1]
        
        #わかってないわかってないわかってないわかってない
        for j, meshnp_px3 in enumerate(meshnp_bxpx3):
            catname, md5name, numname = da['cate'][j], da['md5'][j], da['view0']['num'][j]
            mesh_dir = os.path.join(svfolder, 'test-%d' % model_iter,
                                    '{}/{}/{}.obj'.format(catname, md5name, numname))
            if not os.path.exists(os.path.join(svfolder, 'test-%d' % model_iter, catname, md5name)):
                os.makedirs(os.path.join(svfolder, 'test-%d' % model_iter, catname, md5name))
            tmo = meshnp_px3
            savemeshcolor(tmo, facenp_fx3, mesh_dir, meshcolnp_bxpx3[j])    
    
    #最終的な損失
    re = []
    for ca in cates:
        iou[ca] /= catenum[ca]
        print('{}, {}'.format(ca, iou[ca]))
        re.append([int(ca), iou[ca]])
    re = np.array(re, dtype=np.float32)
    path = os.path.join(svfolder, 'test-%d.npy' % test_iter_num)
    np.save(file=path, arr=re)    
            
        
if __name__ == '__main__':
    test()

