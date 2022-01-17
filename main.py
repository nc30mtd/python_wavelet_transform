import numpy as np
import matplotlib.pyplot as plt
import ta_wavelet

if __name__ == "__main__":

    # PyWaveletsで使用できるウェーブレット波形の種類は以下で確認できます
    # wavlist = pywt.wavelist(kind='continuous')
    # print(wavlist)

    dt = 0.01 # サンプリング間隔（時間）
    # fs = 1/dt # サンプリング周波数
    # nq_f = fs/2.0 # ナイキスト周波数
    wavelet_type = 'cmor1.5-1.0'

    signal = ta_wavelet.get_src_data(dt)
    cwtmatr, absmat = ta_wavelet.get_wavelet_data(signal, dt, wavelet_type)

    print(signal.shape)
    print(cwtmatr.shape)
    print(absmat)
    print(np.max(absmat))
    print(np.min(absmat))

    normalized_absmat = ta_wavelet.minmax_norm(absmat)
    print(normalized_absmat)
    print(np.max(normalized_absmat))
    print(np.min(normalized_absmat))

    plt.imshow(normalized_absmat, aspect='auto')
    plt.colorbar()
    plt.show()