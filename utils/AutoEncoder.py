from __future__ import print_function
from __future__ import division

#pytorch内に定義されている，nn(neural network)に関するモジュール
import torch
import torch.nn as nn 
import torch.nn.functional as F


class Ecoder(nn.Module):  #nn.Moduleを継承している
    def __init__(self, N_CHANNELS, N_KERNELS, BATCH_SIZE,IMG_DIM,VERBOSE = False, pred_cam = False):  #コンストラクタであり初めてclassを呼び出したときに実行される
        
        #nn.Moduleのコンストラクタを呼び出している
        super(Ecoder, self).__init__()  
        
        
        #以下エンコーダー用のlayer定義
        #畳み込み層の定義(RGB画像なので4チャンネルのフィルターの数を64/128/256と増やす)
        block1 = self.convblock(N_CHANNELS, 64, N_KERNELS,stride = 2, pad = 2)
        block2 = self.convblock(64, 128, N_KERNELS,stride = 2, pad = 2)
        block3 = self.convblock(128, 256, N_KERNELS,stride = 2, pad = 2)
        
        #線形層の関数(1，2層目にバッチ正規化と活性化関数ReLUを適用．各層には1024個のニューロン)
        #畳み込み演算で8*8の画像が256チャンネルでき，8*8*256の16384にフラット化
        linear1 = self.linearblock(16384, 1024)
        linear2 = self.linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        
        #????????????????????????????????????????????????????????????????????????
        self.pred_cam = pred_cam
        if self.pred_cam:
            self.pred_cam_linear_1 = nn.Linear(1024, 128)
            self.pred_cam_linear_2 = nn.Linear(128, 9 + 3)
            
            
        #以下デコーダー用のlayer定義
        #頂点用とカラー用のデコーダーがあり，各レイヤーには1024/2048/1926ニューロンが存在
        linear4 = self.linearblock(1024, 1024)
        linear5 = self.linearblock(1024, 2048)
        self.linear6 = nn.Linear(2048, 1926)
        
        linear42 = self.linearblock(1024, 1024)
        linear52 = self.linearblock(1024, 2048)
        self.linear62 = nn.Linear(2048, 1926)
        
        #以下各レイヤーの結合し，エンコーダー/デコーダーの定義
        #Sequential：層のリストを形成
        all_blocks = block1 + block2 + block3
        self.encoder1 = nn.Sequential(*all_blocks)
        
        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)
        
        all_blocks = linear4 + linear5
        self.decoder = nn.Sequential(*all_blocks)
        
        all_blocks = linear42 + linear52
        self.decoder2 = nn.Sequential(*all_blocks)
        
        #Xaxier(人の名前)の重みの初期化
        #isinstanceは方の判定を行う(True/False)
        #self.modulesとobjectが??????????????????????????????????????????????????????
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)  #標準偏差0.001に正規化
        
        #エンコーダやデコーダに定義は済んでいるので，余計なオブジェクトを削除(del)する
        del all_blocks, block1, block2, block3, \
        linear1, linear2, linear4, linear5, \
        linear42, linear52
        
        #summary：概要を出力する関数
        if VERBOSE:
            self.summary(BATCH_SIZE, N_CHANNELS, IMG_DIM)
        
    #オリジナル畳み込み層の関数
    def convblock(self, indim, outdim, ker, stride, pad):
        block2 = [
            #torch.nn内のモジュール
            nn.Conv2d(indim, outdim, ker, stride, pad), #畳み込み層(1,2:入力と出力のチャンネル数) 
            nn.BatchNorm2d(outdim),   #バッチ正規化
            nn.ReLU()                 #活性化関数ReLu 
        ]
        return block2
    
    #オリジナル線形層
    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2
    
    #順伝播
    def forward(self, x):
        
        for layer in self.encoder1:
            x = layer(x)
        
        #bnumはバッチサイズを表しており，view(bnum,-1)とすることで残りの要素を1次元化している(reshape)
        bnum = x.shape[0] 
        x = x.view(bnum, -1)
        
        for layer in self.encoder2:
            x = layer(x)
        x = self.linear3(x)
        
        if self.pred_cam:
            cam_x = TorchF.relu(self.pred_cam_linear_1(x))
            pred_cam = self.pred_cam_linear_2(cam_x)
        
        x1 = x
        for layer in self.decoder:
            x1 = layer(x1)
        x1 = self.linear6(x1)
        
        x2 = x
        for layer in self.decoder2:
            x2 = layer(x2)
        x2 = self.linear62(x2)
        
        if self.pred_cam:
            return x1, x2, pred_cam
        return x1, x2
    
    #summary：概要を出力する関数
    def summary(self, BATCH_SIZE, N_CHANNELS, IMG_DIM):
        
        x = torch.zeros(BATCH_SIZE, N_CHANNELS, IMG_DIM, IMG_DIM)
        
        # Print the title in a good design
        # for easy recognition.
        print()
        summary_title = '| {} Summary |'.format(self.__class__.__name__)
        for _ in range(len(summary_title)):
            print('-', end='')
        print()
        print(summary_title)
        for _ in range(len(summary_title)):
            print('-', end='')
        print('\n')
       
        # 各層に対して，サイズと層の種類を出力
        #pytorchではfoward計算時に勾配計算用のパラメータを保存しておくことでbackward計算の高速化をしている
        #今回はサイズを出力したいだけなので，余分な保存を行わないようにtorch.no_grad()を用いている
        print('Input: {}'.format(x.size()))
        with torch.no_grad():
            
            print('encoder')
            for layer in self.encoder1:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            
            bnum = x.shape[0] 
            x = x.view(bnum, -1)  # flatten the encoder1 output
            print('Out: {} \tlayer: {}'.format(x.size(), 'Reshape: Flatten'))
            
            for layer in self.encoder2:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer)) 
            x = self.linear3(x)
            print('Out: {} \tLayer: {}'.format(x.size(), self.linear3))
            
            print('decoder')
            for layer in self.decoder:
                x = layer(x)
                print('Out: {} \tLayer: {}'.format(x.size(), layer))
            x = self.linear6(x)
            print('Out: {} \tLayer: {}'.format(x.size(), self.linear6))
    
#importされただけで勝手にmain文が実行されないようにするおまじない
if __name__ == '__main__':

    BATCH_SIZE = 64  #バッチサイズ
    IMG_DIM = 64     #画像サイズ
    N_CHANNELS = 4   #RGBAの4(入力サイズとして用いる)
    N_KERNELS = 5    #カーネルサイズ
    VERBOSE = False  #summary(概要)を出力するか否か
    
    model = Ecoder(N_CHANNELS, N_KERNELS, BATCH_SIZE, IMG_DIM, VERBOSE)#
