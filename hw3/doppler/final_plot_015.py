"""
多普勒效应音频分析程序

本程序用于分析带有多普勒效应的音频信号，通过频谱分析提取频率随时间的变化特征，
并进行周期折叠分析，最终估算声源的运动参数。

作者: CHEN Jingxu, CHEN Yuan
日期: 2025 0419
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt, stft, hilbert
from scipy.optimize import curve_fit
import platform
from mpl_toolkits.mplot3d import Axes3D
import os
import logging
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_matplotlib_fonts():
    """设置matplotlib字体，确保中文显示正常"""
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
    else:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统使用Arial Unicode MS
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def load_audio(file_path):
    """
    加载音频文件
    
    参数:
        file_path (str): 音频文件路径
        
    返回:
        tuple: (采样率, 音频数据)
    """
    logger.info(f"加载音频文件: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到音频文件: {file_path}")
        
    sample_rate, audio = wav.read(file_path)
    logger.info(f"音频采样率: {sample_rate} Hz")
    return sample_rate, audio

def apply_bandpass_filter(audio, sample_rate, lowcut, highcut, order=3):
    """
    应用带通滤波器
    
    参数:
        audio (array): 音频数据
        sample_rate (int): 采样率
        lowcut (float): 低截止频率
        highcut (float): 高截止频率
        order (int): 滤波器阶数
        
    返回:
        array: 滤波后的音频
    """
    logger.info(f"应用带通滤波器 [{lowcut}-{highcut}] Hz, 阶数={order}")
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    filtered_audio = filtfilt(b, a, audio)
    return filtered_audio

# 新增：包络平滑校正
def envelope_correction(signal, sample_rate, smooth_cutoff=40.0, epsilon=1e-8):
    """包络平滑，将信号幅度归一化"""
    analytic = hilbert(signal)
    raw_env = np.abs(analytic)
    b, a = butter(2, smooth_cutoff/(sample_rate/2), 'low')
    smooth_env = filtfilt(b, a, raw_env)
    corrected = signal / (smooth_env + epsilon)
    return corrected, smooth_env

def perform_stft(audio, sample_rate, window_length_sec, overlap_ratio):
    """
    执行短时傅里叶变换 (STFT)
    
    参数:
        audio (array): 音频数据
        sample_rate (int): 采样率
        window_length_sec (float): 窗口长度(秒)
        overlap_ratio (float): 重叠比例 (0-1之间)
        
    返回:
        tuple: (频率数组, 时间数组, STFT结果)
    """
    logger.info(f"执行STFT分析: 窗口长度={window_length_sec}秒, 重叠比例={overlap_ratio}")
    nperseg = int(window_length_sec * sample_rate)
    noverlap = int(nperseg * overlap_ratio)
    return stft(audio, fs=sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap)

def fold_spectrogram_by_period(f, t, Zxx, period):
    """
    将频谱图按周期折叠
    
    参数:
        f (array): 频率数组
        t (array): 时间数组
        Zxx (array): STFT结果
        period (float): 折叠周期(秒)
        
    返回:
        tuple: (相位网格, 频率数组, 折叠后的频谱)
    """
    logger.info(f"将频谱按周期 {period} 秒进行折叠")
    
    # 计算每个时间点对应的相位
    phase = (t % period) / period  # 归一化相位 (0~1)
    phase_in_sec = phase * period  # 转换回秒
    
    # 创建均匀的相位网格
    phase_bins = 100
    phase_grid = np.linspace(0, period, phase_bins)
    freq_bins = len(f)
    folded_spectrogram = np.zeros((freq_bins, phase_bins))
    
    # 对每个时间点的频谱进行叠加
    for i, p in enumerate(phase_in_sec):
        # 找到最接近的相位bin
        bin_idx = np.argmin(np.abs(phase_grid - p))
        # 叠加频谱（累加幅度）
        folded_spectrogram[:, bin_idx] += np.abs(Zxx[:, i])
    
    return phase_grid, f, folded_spectrogram

def calculate_spectral_centroids(f, folded_spectrogram):
    """
    计算折叠频谱的频谱质心
    
    参数:
        f (array): 频率数组
        folded_spectrogram (array): 折叠后的频谱
        
    返回:
        array: 每个相位点的主频率
    """
    logger.info("计算频谱质心...")
    phase_bins = folded_spectrogram.shape[1]
    folded_peak_freqs = []
    
    for i in range(phase_bins):
        spectrum = folded_spectrogram[:, i]
        # 使用频谱质心计算主频率，而不是简单找最大值
        weighted_sum = np.sum(f * spectrum)
        total_sum = np.sum(spectrum)
        
        if total_sum > 0:  # 避免除零错误
            centroid = weighted_sum / total_sum
            folded_peak_freqs.append(centroid)
        else:
            folded_peak_freqs.append(500)  # 默认中心频率
            
    return np.array(folded_peak_freqs)

def plot_spectrogram(phase_grid, f, folded_spectrogram, freq_range=(480, 520)):
    """绘制折叠后的时频图，用亮度表示强度"""
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(phase_grid, f, folded_spectrogram, shading='auto')
    plt.colorbar(label='累积幅度')
    plt.title('多普勒效应周期叠加分析 (T=0.2s)')
    plt.xlabel('周期内时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.ylim(freq_range)
    plt.tight_layout()
    return plt.gcf()

def plot_spectral_centroid_curve(phase_grid, folded_peak_freqs):
    """绘制叠加后的频率随相位变化曲线"""
    plt.figure(figsize=(12, 4))
    plt.plot(phase_grid, folded_peak_freqs, 'r-', linewidth=2)
    plt.title('周期叠加后的频率变化曲线 (T=0.2s)')
    plt.xlabel('周期内时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.grid(True)
    plt.ylim(np.min(folded_peak_freqs) - 2, np.max(folded_peak_freqs) + 2)
    plt.tight_layout()
    return plt.gcf()

def plot_3d_spectrogram(phase_grid, f, folded_spectrogram, freq_range=(480, 520)):
    """绘制3D频谱图"""
    logger.info("绘制3D频谱图...")
    try:
        # 创建相位和频率的网格
        mask = (f <= freq_range[1]) & (f >= freq_range[0])
        phase_mesh, freq_mesh = np.meshgrid(phase_grid, f[mask])
        
        # 截取感兴趣的频率范围
        intensity = folded_spectrogram[mask, :]
        
        # 3D绘图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 使用曲面图显示
        surf = ax.plot_surface(phase_mesh, freq_mesh, intensity, cmap='viridis',
                              linewidth=0, antialiased=True, alpha=0.8)
        
        ax.set_title('多普勒效应周期频谱的3D表示')
        ax.set_xlabel('周期内时间 (秒)')
        ax.set_ylabel('频率 (Hz)')
        ax.set_zlabel('累积幅度')
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='幅度')
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"3D可视化失败: {e}")
        print(f"3D可视化失败，可能是由于缺少必要的依赖: {e}")
        return None

def calculate_spectral_centroids_direct(audio, sample_rate, window_size, hop_size):
    """
    直接从音频计算频谱质心
    
    参数:
        audio (array): 音频数据
        sample_rate (int): 采样率
        window_size (int): 窗口大小(采样点数)
        hop_size (int): 跳跃大小(采样点数)
        
    返回:
        tuple: (时间数组, 频谱质心数组)
    """
    logger.info(f"直接计算频谱质心: 窗口大小={window_size}, 跳跃大小={hop_size}")
    frequencies = []
    times = []
    
    for start in range(0, len(audio) - window_size, hop_size):
        segment = audio[start:start + window_size]
        windowed = segment * np.hanning(window_size)
        spectrum = np.fft.rfft(windowed)
        freq = np.fft.rfftfreq(window_size, d=1/sample_rate)
        magnitude = np.abs(spectrum)

        spectral_centroid = np.sum(freq * magnitude) / np.sum(magnitude)

        frequencies.append(spectral_centroid)
        times.append(start / sample_rate)
        
    return np.array(times), np.array(frequencies)

def fold_by_period(times, values, period):
    """
    将时间序列数据按指定周期折叠
    
    参数:
        times (array): 时间数组
        values (array): 数值数组
        period (float): 折叠周期(秒)
        
    返回:
        tuple: (折叠后的时间数组, 折叠后的数值数组)
    """
    folded_times = np.array(times) % period
    # 按折叠后的时间排序
    sort_idx = np.argsort(folded_times)
    return folded_times[sort_idx], np.array(values)[sort_idx]

def bin_folded_data(folded_times, folded_freqs, period, num_bins=50):
    """
    对折叠后的数据进行分箱处理
    
    参数:
        folded_times (array): 折叠后的时间
        folded_freqs (array): 折叠后的频率
        period (float): 周期(秒)
        num_bins (int): 分箱数量
        
    返回:
        tuple: (有效的箱中心, 平均频率, 有效箱的掩码)
    """
    bins = np.linspace(0, period, num_bins+1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 计算每个点属于哪个箱
    bin_indices = np.digitize(folded_times, bins)
    
    # 累加并计算平均值
    bin_means = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    
    for i in range(len(folded_times)):
        if 0 < bin_indices[i] <= num_bins:  # 确保索引在范围内
            bin_means[bin_indices[i]-1] += folded_freqs[i]
            bin_counts[bin_indices[i]-1] += 1
    
    # 避免除零
    valid_bins = bin_counts > 0
    bin_means[valid_bins] = bin_means[valid_bins] / bin_counts[valid_bins]
    
    return bin_centers, bin_means, valid_bins

def fourier_series(x, a0, a1, b1, a2, b2, period=0.2):
    """
    简单的傅里叶级数模型 (使用2项谐波)
    
    参数:
        x (array): 自变量
        a0, a1, b1, a2, b2: 傅里叶系数
        period (float): 周期
        
    返回:
        array: 函数值
    """
    f = a0
    f += a1 * np.cos(2 * np.pi * x / period)
    f += b1 * np.sin(2 * np.pi * x / period)
    f += a2 * np.cos(4 * np.pi * x / period)
    f += b2 * np.sin(4 * np.pi * x / period)
    return f

def fit_fourier_series(valid_times, valid_freqs, period):
    """
    使用傅里叶级数拟合频率数据
    
    参数:
        valid_times (array): 时间点
        valid_freqs (array): 对应的频率
        period (float): 周期(秒)
        
    返回:
        tuple: (拟合参数, 拟合曲线的x, 拟合曲线的y)
    """
    try:
        # 初始参数猜测
        initial_guess = [1000, 5, 5, 1, 1]  
        
        # 拟合傅里叶级数
        params, _ = curve_fit(
            lambda x, a0, a1, b1, a2, b2: fourier_series(x, a0, a1, b1, a2, b2, period), 
            valid_times, valid_freqs, p0=initial_guess
        )
        
        # 生成拟合曲线的连续点
        fit_x = np.linspace(0, period, 1000)
        fit_y = fourier_series(fit_x, *params, period)
        
        return params, fit_x, fit_y
    except Exception as e:
        logger.error(f"傅里叶级数拟合失败: {e}")
        return None, None, None

def plot_folded_data(folded_times, folded_freqs, bin_centers, bin_means, 
                     valid_bins, best_period, fit_x=None, fit_y=None, params=None):
    """绘制折叠后的数据散点图和拟合曲线"""
    plt.figure(figsize=(14, 6))
    
    # 散点图显示所有折叠数据
    plt.scatter(folded_times, folded_freqs, alpha=0.3, s=10, c='blue', label='折叠数据点')
    
    # 绘制平均曲线
    plt.plot(bin_centers[valid_bins], bin_means[valid_bins], 'r-', linewidth=3, label='周期内平均频率')
    
    # 如果有拟合曲线，则绘制
    if fit_x is not None and fit_y is not None:
        plt.plot(fit_x, fit_y, 'g-', linewidth=2, label='傅里叶级数拟合')
    
    plt.title(f'周期折叠后的频率变化 (T={best_period:.6f}s)')
    plt.xlabel('周期内时间 (秒)')
    plt.ylabel('频率 (Hz)')
    
    # 根据数据范围设置y轴限制
    y_min = np.min(bin_means[valid_bins]) - 2
    y_max = np.max(bin_means[valid_bins]) + 2
    plt.ylim(y_min, y_max)
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # 如果有拟合参数，则打印
    if params is not None:
        a0, a1, b1, a2, b2 = params
        logger.info(f"傅里叶级数拟合参数:")
        logger.info(f"  a0 (平均值): {a0:.3f}")
        logger.info(f"  一次谐波: a1={a1:.3f}, b1={b1:.3f}, 幅度={np.sqrt(a1**2 + b1**2):.3f}")
        logger.info(f"  二次谐波: a2={a2:.3f}, b2={b2:.3f}, 幅度={np.sqrt(a2**2 + b2**2):.3f}")
    
    return plt.gcf()

def plot_phase_info(bin_centers, bin_means, valid_bins, folded_freqs, period):
    """绘制相位信息和频率统计"""
    plt.figure(figsize=(10, 6))

    # 绘制周期内的频率变化范围
    plt.fill_between(bin_centers[valid_bins], 
                    np.percentile(folded_freqs, 25, interpolation='linear'),
                    np.percentile(folded_freqs, 75, interpolation='linear'),
                    alpha=0.2, color='blue', label='频率25%-75%区间')

    plt.plot(bin_centers[valid_bins], bin_means[valid_bins], 'r-', linewidth=3)

    # 标记最大值和最小值点
    valid_means = bin_means[valid_bins]
    valid_centers = bin_centers[valid_bins]
    
    max_idx = np.argmax(valid_means)
    min_idx = np.argmin(valid_means)

    max_time = valid_centers[max_idx]
    min_time = valid_centers[min_idx]

    plt.scatter(max_time, valid_means[max_idx], color='red', s=100, 
                marker='*', label=f'最高频率: {valid_means[max_idx]:.2f}Hz @ {max_time:.3f}s')
    plt.scatter(min_time, valid_means[min_idx], color='blue', s=100, 
                marker='*', label=f'最低频率: {valid_means[min_idx]:.2f}Hz @ {min_time:.3f}s')

    # 计算频率变化率
    max_freq = valid_means[max_idx]
    min_freq = valid_means[min_idx]
    center_freq = (max_freq + min_freq) / 2
    freq_deviation = (max_freq - min_freq) / 2

    plt.axhline(y=center_freq, color='k', linestyle='--', alpha=0.5)
    plt.text(period*0.01, center_freq+0.5, f'中心频率: {center_freq:.2f}Hz', fontsize=10)

    plt.title(f'多普勒效应周期分析: 中心频率{center_freq:.2f}Hz, 偏移幅度±{freq_deviation:.2f}Hz')
    plt.xlabel('周期内时间 (秒)')
    plt.ylabel('频率 (Hz)')
    plt.ylim(center_freq-freq_deviation*1.5, center_freq+freq_deviation*1.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt.gcf(), max_time, min_time, max_freq, min_freq, center_freq, freq_deviation

def estimate_doppler_parameters(period, center_freq, freq_deviation, max_time, min_time):
    """根据多普勒效应估算运动参数"""
    logger.info("\n多普勒效应分析结果:")
    logger.info(f"周期: {period} 秒")
    logger.info(f"中心频率: {center_freq:.3f} Hz")
    logger.info(f"频率偏移: ±{freq_deviation:.3f} Hz")
    logger.info(f"最大频率出现在周期的 {max_time/period*100:.1f}% 处")
    logger.info(f"最小频率出现在周期的 {min_time/period*100:.1f}% 处")

    # 根据多普勒效应公式估算速度
    # f_observed = f_source * (c + v_observer)/(c - v_source)
    # 假设声源相对静止的观察者做简谐运动
    c = 343  # 声速，m/s
    f_source = center_freq  # 声源频率，Hz

    # 频率变化与速度的关系: Δf/f ≈ v/c (当v<<c时的近似)
    velocity_amplitude = c * freq_deviation / f_source
    displacement_amplitude = velocity_amplitude / (2 * np.pi / period)

    logger.info(f"\n估算的速度振幅: {velocity_amplitude:.3f} m/s")
    logger.info(f"对应的位移振幅: {displacement_amplitude:.3f} m")
    
    return velocity_amplitude, displacement_amplitude

# 新增：在误差范围内搜索最佳周期
def optimize_period(times, freqs, initial_period, search_frac=0.01, num=1001):
    best_period = initial_period
    best_res = np.inf
    best_params = best_fx = best_fy = None
    periods = np.linspace(initial_period*(1-search_frac),
                          initial_period*(1+search_frac), num)
    for p in periods:
        # 1. 折叠
        ft = np.array(times) % p
        # 2. 分箱得质心光谱曲线
        bin_centers, bin_means, valid = bin_folded_data(ft, freqs, p)
        # 3. 在每个折叠时刻上用该曲线插值出"估计频率"
        #    注意使用原始顺序 times 对齐 freqs
        est_freq = np.interp(ft, bin_centers[valid], bin_means[valid])
        # 4. 计算残差
        res = np.sum((freqs - est_freq)**2)
        if res < best_res:
            best_res, best_period = res, p
    # 不再返回拟合结果，保持签名兼容
    return best_period, None, None, None

def main():
    """主函数，执行整个分析流程"""
    # 设置matplotlib字体
    setup_matplotlib_fonts()
    
    # 1. 定义参数
    audio_file = 'rotation_doppler_simulation.wav'
    audio_file = 'duopule/output/14_output.wav' #period = 0.2003会好
    audio_file = '(66,30)_(6,25)_4s_85.236d_site1.wav'
    # audio_file = '250418/output/REC_015_output.wav' # 17有点古怪。18+window_length_sec = 0.049+period = 0.2003结果比较好。14和15也有点怪。
    lowcut = 460.0
    highcut = 540.0
    window_length_sec = 1.99  # 可以直接看到理论和实际是基本上完全一致的
    period = 4
    
    # 2. 加载音频文件
    sample_rate, audio = load_audio(audio_file)
    
    # 3. 带通滤波
    filtered_audio = apply_bandpass_filter(audio, sample_rate, lowcut, highcut)
    # 新增：包络平滑校正处理
    filtered_audio, smooth_env = envelope_correction(
        filtered_audio, sample_rate, smooth_cutoff=40.0)
    # 4. 短时傅里叶变换 (STFT)
    overlap_ratio = 0.99
    f, t, Zxx = perform_stft(filtered_audio, sample_rate, window_length_sec, overlap_ratio)
    
    # 5. 按周期折叠频谱
    phase_grid, _, folded_spectrogram = fold_spectrogram_by_period(f, t, Zxx, period)
    
    # 6. 计算频谱质心
    folded_peak_freqs = calculate_spectral_centroids(f, folded_spectrogram)
    
    # 7. 绘制时频图
    # fig1 = plot_spectrogram(phase_grid, f, folded_spectrogram)
    # plt.show()
    
    # 8. 绘制频率随相位变化曲线
    # fig2 = plot_spectral_centroid_curve(phase_grid, folded_peak_freqs)
    # plt.show()
    
    # 9. 计算频率变化的统计信息
    freq_mean = np.mean(folded_peak_freqs)
    freq_amplitude = (np.max(folded_peak_freqs) - np.min(folded_peak_freqs)) / 2
    max_freq_idx = np.argmax(folded_peak_freqs)
    max_freq_phase = phase_grid[max_freq_idx]
    
    print(f"中心频率: {freq_mean:.2f} Hz")
    print(f"频率变化幅度: ±{freq_amplitude:.2f} Hz")
    print(f"频率最高点出现在周期内时间点: {max_freq_phase:.3f} 秒")
    
    # 10. 3D可视化
    # fig3 = plot_3d_spectrogram(phase_grid, f, folded_spectrogram)
    # if fig3:
    #     plt.show()
    
    # 11. 直接从音频计算频谱质心
    nperseg = int(window_length_sec * sample_rate)
    hop_size = nperseg // 16
    times, frequencies = calculate_spectral_centroids_direct(filtered_audio, sample_rate, nperseg, hop_size)
    
    # 新增：优化周期
    best_period, opt_params, opt_fx, opt_fy = optimize_period(
        times, frequencies, period)
    print(f"优化后最佳周期: {best_period:.6f} 秒")
    
    # 12. 折叠原始频谱质心数据 (使用最佳周期)
    folded_times, folded_freqs = fold_by_period(times, frequencies, best_period)
    
    # 13. 对数据进行分箱处理
    bin_centers, bin_means, valid_bins = bin_folded_data(folded_times, folded_freqs, best_period)
    
    # 15. 绘制折叠数据和拟合曲线
    fig4 = plot_folded_data(folded_times, folded_freqs, bin_centers, bin_means, 
                           valid_bins, best_period, params=opt_params)
    plt.show()
    
    # 16. 计算并绘制相位信息
    fig5, max_time, min_time, max_freq, min_freq, center_freq, freq_deviation = plot_phase_info(
        bin_centers, bin_means, valid_bins, folded_freqs, best_period)
    plt.show()
    
    # 17. 估算多普勒效应参数
    velocity_amplitude, displacement_amplitude = estimate_doppler_parameters(
        best_period, center_freq, freq_deviation, max_time, min_time)
    
    print(f"\n多普勒效应分析结果:")
    print(f"周期: {best_period} 秒")
    print(f"中心频率: {center_freq:.3f} Hz")
    print(f"频率偏移: ±{freq_deviation:.3f} Hz")
    print(f"最大频率出现在周期的 {max_time/best_period*100:.1f}% 处")
    print(f"最小频率出现在周期的 {min_time/best_period*100:.1f}% 处")
    print(f"\n估算的速度振幅: {velocity_amplitude:.3f} m/s")
    print(f"对应的位移振幅: {displacement_amplitude:.3f} m")

def main2():
    setup_matplotlib_fonts()
    
    # 支持多个参数及包络平滑开关
    parser = argparse.ArgumentParser(description='多普勒效应参数比较')
    parser.add_argument('--window_lengths', type=str, default='0.01, 0.025, 0.05,0.075, 0.1,0.15, 0.2, 0.3, 0.4, 0.6')
    parser.add_argument('--overlap_ratios', type=str, default='0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9')
    parser.add_argument('--smooth_cutoff', type=float, default=40.0,
                        help='包络平滑低通截止频率(Hz)')
    args = parser.parse_args()
    window_lengths = [float(x) for x in args.window_lengths.split(',')]
    overlap_ratios = [float(x) for x in args.overlap_ratios.split(',')]

    audio_file = 'rotation_doppler_simulation.wav'
    audio_file = 'duopule/output/12_output.wav' #period = 0.2003会好
    lowcut, highcut = 960.0, 1040.0
    period = 0.200317

    sample_rate, audio = load_audio(audio_file)
    filtered_audio = apply_bandpass_filter(audio, sample_rate, lowcut, highcut)

    # 初始化偏移矩阵：行=window_lengths, 列=overlap_ratios
    deviations = np.zeros((len(window_lengths), len(overlap_ratios)))

    for i, wl in enumerate(window_lengths):
        for j, oratio in enumerate(overlap_ratios):
            f, t, Zxx = perform_stft(filtered_audio, sample_rate, wl, oratio)
            phase_grid, _, folded_spec = fold_spectrogram_by_period(f, t, Zxx, period)
            freqs = calculate_spectral_centroids(f, folded_spec)

            mean_f = np.mean(freqs)
            dev_f  = (np.max(freqs) - np.min(freqs)) / 2
            # 存储到矩阵
            deviations[i, j] = dev_f
            print(f'窗口={wl:.3f}s, 重叠={oratio:.2f} -> '
                  f'中心频率={mean_f:.2f}Hz, 偏移=±{dev_f:.2f}Hz')

    # 绘制二维热图：window_lengths vs overlap_ratios
    plt.figure(figsize=(8, 6))
    im = plt.imshow(deviations, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label='频率偏移 ±Hz')
    plt.xticks(ticks=range(len(overlap_ratios)),
               labels=[f'{x:.2f}' for x in overlap_ratios])
    plt.yticks(ticks=range(len(window_lengths)),
               labels=[f'{x:.3f}' for x in window_lengths])
    plt.xlabel('重叠比例')
    plt.ylabel('窗口长度 (秒)')
    plt.title('不同窗口长度与重叠比例下的频率偏移热图')
    plt.tight_layout()
    plt.show()

    return

if __name__ == "__main__":
    main()