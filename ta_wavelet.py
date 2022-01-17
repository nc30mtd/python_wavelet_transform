import numpy as np
import matplotlib.pyplot as plt
import pywt


def get_src_data(dt):
    t = np.arange(-1, 1, dt)
    sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))

    # plt.plot(t, sig)
    # plt.show()

    return sig

def get_scales(nq_f, fs):
    # 解析したい周波数のリスト（ナイキスト周波数以下）
    # 1 Hz ～ nq_f Hzの間を等間隔に50分割
    freqs = np.linspace(1,nq_f,50)

    # サンプリング周波数に対する比率を算出
    freqs_rate = freqs / fs

    # スケール：サンプリング周波数＝1:fs(1/dt)としてスケールに換算
    scales = 1 / freqs_rate
    # 逆順に入れ替え
    scales = scales[::-1]
    #print(scales)

    return scales

# 
# ウェーブレットの形状を確認する
# 
def check_wavelet_type(wavelet_type):
    #wavelet_type = 'cmor1.5-1.0'
    wav = pywt.ContinuousWavelet(wavelet_type)

    # precisionによってデータ個数(len)が変わる
    int_psi, x = pywt.integrate_wavelet(wav, precision=8)

    #print(len(int_psi))
    plt.plot(x, int_psi)

    plt.show()

#
# 二次元配列を正規化
#
def normalize(v, axis=-1, order=2):
    l2 = np.linalg.norm(v, ord = order, axis=axis, keepdims=True)
    l2[l2==0] = 1
    return v/l2
#
# 指定した列のデータを0～1に正規化する
#
def minmax_norm(arr):
    # scaler = MinMaxScaler()
    # df.loc[:,:] = scaler.fit_transform(df)
    # return df
    return (arr - np.min(arr)) / ( np.max(arr) - np.min(arr))

def get_wavelet_data(signal, dt, wavelet_type):
    fs = 1/dt # サンプリング周波数
    nq_f = fs/2.0 # ナイキスト周波数

    #sig = get_src_data(dt)
    scales = get_scales(nq_f, fs)

    cwtmatr, freqs_rate = pywt.cwt(signal, scales=scales, wavelet=wavelet_type)
    absmat = np.abs(cwtmatr)

    return cwtmatr, absmat
