import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import pywt
import platform
import os
import logging
from scipy.signal import butter, filtfilt

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_matplotlib_fonts():
    """设置matplotlib字体，确保中文显示正常"""
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

def load_audio(file_path):
    """加载音频文件"""
    logger.info(f"加载音频文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到音频文件: {file_path}")
    sample_rate, audio = wav.read(file_path)
    logger.info(f"音频采样率: {sample_rate} Hz")
    return sample_rate, audio

def apply_bandpass_filter(audio, sample_rate, lowcut, highcut, order=3):
    """应用带通滤波器"""
    logger.info(f"应用带通滤波器 [{lowcut}-{highcut}] Hz, 阶数={order}")
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)
    return filtered

def perform_cwt(audio, sample_rate, wavelet='cmor2-5', max_scale=128):
    """连续小波变换 (CWT)，优化尺度映射到 900–1100Hz"""
    # 根据目标频率范围生成等间距频率，再映射为尺度
    fc = pywt.central_frequency(wavelet)
    freqs_target = np.linspace(1040, 960, max_scale)
    scales = fc * sample_rate / freqs_target
    coefs, freqs = pywt.cwt(audio, scales, wavelet, sampling_period=1/sample_rate)
    power = np.abs(coefs) ** 2
    return scales, freqs, power

def plot_scalogram(power, freqs, times, freq_range=(900, 1100)):
    """绘制小波时频图（scalogram）"""
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, freqs, power, shading='auto')
    plt.colorbar(label='功率')
    plt.title('连续小波变换时频图')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.ylim(freq_range)
    plt.tight_layout()
    return plt.gcf()

# 新增：按周期折叠小波功率
def fold_spectrogram_by_period(freqs, power, times, period, phase_bins=100):
    logger.info(f"将小波功率按周期 {period}s 折叠")
    phase = (times % period) / period * period
    phase_grid = np.linspace(0, period, phase_bins)
    folded = np.zeros((len(freqs), phase_bins))
    for i, p in enumerate(phase):
        idx = np.argmin(np.abs(phase_grid - p))
        folded[:, idx] += power[:, i]
    return phase_grid, freqs, folded

# 新增：计算折叠频谱的质心
def calculate_spectral_centroids(freqs, folded):
    logger.info("计算折叠后频谱质心")
    centroids = []
    for i in range(folded.shape[1]):
        spec = folded[:, i]
        total = spec.sum()
        centroids.append((freqs * spec).sum() / total if total > 0 else freqs.mean())
    return np.array(centroids)

# 新增：绘制质心随相位变化曲线
def plot_spectral_centroid_curve(phase_grid, centroids):
    plt.figure(figsize=(12, 4))
    plt.plot(phase_grid, centroids, 'r-', lw=2)
    plt.title('折叠后频谱质心变化')
    plt.xlabel('周期内时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()

def main():
    setup_matplotlib_fonts()
    audio_file = 'duopule\\11.wav'
    lowcut, highcut = 980.0, 1020.0

    # 加载 & 带通滤波
    sample_rate, audio = load_audio(audio_file)
    filtered = apply_bandpass_filter(audio, sample_rate, lowcut, highcut)

    # 连续小波变换 & 绘图
    scales, freqs, power = perform_cwt(filtered, sample_rate, wavelet='cmor5-5', max_scale=64)
    times = np.arange(len(filtered)) / sample_rate
    fig1 = plot_scalogram(power, freqs, times, freq_range=(960, 1040))
    plt.xlim(3, 3.6)
    plt.show()

    # 新增：周期折叠 & 质心分析
    period = 0.2  # 根据实际信号周期设置
    phase_grid, freqs, folded = fold_spectrogram_by_period(freqs, power, times, period)
    centroids = calculate_spectral_centroids(freqs, folded)
    fig2 = plot_spectral_centroid_curve(phase_grid, centroids)
    plt.show()

if __name__ == "__main__":
    main()
