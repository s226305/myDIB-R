#ArgumentParser：変数名，型，初期値を記録するのに用いる
import argparse

def get_args():
    #ArgumentParserを宣言し，説明としてdab-rを加える
    parser = argparse.ArgumentParser(description='dib-r')
    
    #4つの引数があり前から順に，変数名，型，初期値，この変数の説明(なくてもよい)
    #ファイルリスト(ここではデータセット)の名前
    parser.add_argument('--filelist', type=str, default='test_list.txt', help='filelist name')
    
    #pytirchのDataLoaderを使う際に複数処理(いくつのコア)でデータをロードするかを指定
    parser.add_argument('--thread', type=int, default=8, help='num of workers')
    
    #実験用セーブフォルダー
    parser.add_argument('--svfolder', type=str, default='prediction', help='save folder for experiments ')
    
    #事前トレーニングデータ(どこにあるかわかっていない)
    parser.add_argument('--g_model_dir', type=str, default='checkpoints/g_model.pth', help='save path for pretrained model')
                        
    #データセット
    parser.add_argument('--data_folder', type=str, default='dataset', help='data folder')
    
    #反復のスタート
    parser.add_argument('--iter', type=int, default=-1, help='start iteration')
    
    #損失の種類
    parser.add_argument('--loss', type=str, default='iou', help='loss type')
                        
    #カメラモード(perが何かはわかっていない)
    parser.add_argument('--camera', type=str, default='per',help='camera mode')
                        
    #カメラの数
    parser.add_argument('--view', type=int, default=2,help='view number')
                        
    #画像サイズ
    parser.add_argument('--img_dim', type=int, default=64,help='dim of image')
    
    #チャンネル数(RGBAの4チャンネル)
    parser.add_argument('--img_channels', type=int, default=4,help='image channels')
    
    #バッチサイズ
    parser.add_argument('--batch_size', type=int, default=64,help='batch size')
    
    #エポック数
    parser.add_argument('--epoch', type=int, default=1000,help='training epoch')
    
    #ログごとの反復
    parser.add_argument('--iter_log', type=int, default=50,help='iterations per log')
   
    #サンプルごとの反復
    parser.add_argument('--iter_sample', type=int, default=1000,help='iterations per sample')
    
    #モデルの保存ごとの反復
    parser.add_argument('--iter_model', type=int, default=10000,help='iterations per model saving')
    
    #ハイパーパラメーター(silが何かはわかってない)
    parser.add_argument('--sil_lambda', type=float, default=1,help='hyperparamter for sil')
    
    args = parser.parse_args()

    return args

    
    
