import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import utils # For cpu_POS
import config

def generate_final_plots(buffers, cam_params, display_config_not_used): # display_config not used here
    fs = cam_params['fs']
    eps = cam_params['eps']
    min_frames_for_pos_output = int(1.6*fs) + int(fs*1) # Should be from config or buffers init
    min_frames_for_resp_output = int(fs*5) # Should be from config or buffers init

    # rPPG Plot
    if len(buffers['g_signal_list']) > min_frames_for_pos_output:
        print("Plotting final rPPG signals...")
        final_r,final_g,final_b = np.array(list(buffers['r_signal_list'])),np.array(list(buffers['g_signal_list'])),np.array(list(buffers['b_signal_list']))
        final_pos_full = utils.cpu_POS(np.array([final_r,final_g,final_b]).reshape(1,3,-1),fps=fs)[0]
        times_rppg_full = np.array(buffers['collected_rppg_timestamps'][:len(final_pos_full)])
        
        plt.figure(figsize=(15,8)); plt.suptitle("rPPG (POS) Full Session Analysis", fontsize=14)
        plt.subplot(211); plt.plot(times_rppg_full,final_pos_full, label="Raw POS"); plt.title("Raw POS Signal"); plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
        
        det_pos_full = signal.detrend(final_pos_full)
        nyq = 0.5*fs
        low = config.RPPG_POS_LOWCUT_HZ / nyq
        high = config.RPPG_POS_HIGHCUT_HZ / nyq
        avg_bpm_str = "Avg. HR: N/A"
        
        if low > 0 and high < 1 and low < high:
            bf,af = signal.butter(config.RPPG_POS_FILTER_ORDER, [low,high], btype='band')
            filt_pos_full = signal.filtfilt(bf,af,det_pos_full)
            norm_filt_pos_full = (filt_pos_full-np.mean(filt_pos_full))/(np.std(filt_pos_full)+eps)
            peaks_hr, _ = signal.find_peaks(norm_filt_pos_full, prominence=config.RPPG_PEAK_PROMINENCE, distance=fs/(config.RPPG_POS_HIGHCUT_HZ*2.0))
            
            plt.subplot(212)
            plt.plot(times_rppg_full, filt_pos_full, label=f"Filtered POS ({config.RPPG_POS_LOWCUT_HZ}-{config.RPPG_POS_HIGHCUT_HZ}Hz)", color='orange')
            
            if len(peaks_hr) > 0:
                plt.plot(times_rppg_full[peaks_hr], filt_pos_full[peaks_hr], "bx", label="Detected Heartbeats")
                if len(peaks_hr) >= 2:
                    avg_hr_val = 60.0 / (np.mean(np.diff(peaks_hr)) / fs)
                    avg_bpm_str = f"Avg. HR: {avg_hr_val:.1f} BPM"
            plt.title(f"Filtered POS Signal - {avg_bpm_str}")
        else:
            plt.subplot(212);plt.plot(times_rppg_full, det_pos_full, label="Detrended POS (Filter Error)", color='red')
            plt.title(f"Detrended POS Signal - {avg_bpm_str}")
            
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(config.FINAL_RPPG_PLOT_FILENAME); plt.show(block=False)

    # Respiration Plot
    if len(buffers['respiration_of_raw_signal_list']) > min_frames_for_resp_output:
        print("Plotting final Respiration (Shoulder Movement) signals...")
        final_resp_of_full = np.array(list(buffers['respiration_of_raw_signal_list']))
        times_resp_full = np.array(buffers['collected_respiration_of_timestamps'][:len(final_resp_of_full)])
        
        plt.figure(figsize=(15,8))
        plt.suptitle("Respiration from Shoulder Movement (Lucas-Kanade Optical Flow) Analysis", fontsize=14)
        plt.subplot(211); plt.plot(times_resp_full, final_resp_of_full, color='r', label="Raw Shoulder Movement"); plt.title("Raw Respiration Signal from Shoulder Movement"); plt.xlabel("Time (s)"); plt.ylabel("Avg. Y Pixel Position"); plt.legend(); plt.grid(True)
        
        det_resp_full_of = signal.detrend(final_resp_of_full)
        window_size_med_full = int(fs * config.RESP_OF_MEDIAN_FILTER_WINDOW_SECONDS)
        if window_size_med_full % 2 == 0: window_size_med_full += 1
        if window_size_med_full < 3: window_size_med_full = 3
        med_filt_resp_full = signal.medfilt(det_resp_full_of, kernel_size=window_size_med_full)

        nyq_r = 0.5*fs
        low_r = config.RESP_OF_LOWCUT_HZ / nyq_r
        high_r = config.RESP_OF_HIGHCUT_HZ / nyq_r
        avg_rpm_str = "Avg. Rate: N/A"
        
        if low_r > 0 and high_r < 1 and low_r < high_r:
            br,ar = signal.butter(config.RESP_OF_FILTER_ORDER, [low_r,high_r], btype='band')
            filt_resp_full_of = signal.filtfilt(br,ar,med_filt_resp_full)
            norm_sig_resp_full = (filt_resp_full_of-np.mean(filt_resp_full_of))/(np.std(filt_resp_full_of)+eps)
            signal_amplitude_resp_full = np.abs(norm_sig_resp_full)
            adaptive_prominence_resp_full = np.mean(signal_amplitude_resp_full) * config.RESP_OF_PEAK_PROMINENCE_FACTOR
            prominence_threshold_resp_full = max(adaptive_prominence_resp_full, config.RESP_OF_MIN_PEAK_PROMINENCE)

            peaks_rpm, _ = signal.find_peaks(norm_sig_resp_full, 
                                             prominence=prominence_threshold_resp_full, 
                                             distance=fs/(config.RESP_OF_HIGHCUT_HZ*2.5),
                                             width=(int(fs*config.RESP_OF_PEAK_WIDTH_MIN_SECONDS), int(fs*config.RESP_OF_PEAK_WIDTH_MAX_SECONDS)))
            
            plt.subplot(212)
            plt.plot(times_resp_full, filt_resp_full_of, color='g', label=f"Filtered Shoulder Movement ({config.RESP_OF_LOWCUT_HZ}-{config.RESP_OF_HIGHCUT_HZ}Hz)")
            
            if len(peaks_rpm) > 0:
                plt.plot(times_resp_full[peaks_rpm], filt_resp_full_of[peaks_rpm], "rx", label="Detected Breaths")
                if len(peaks_rpm) >= 2:
                    peak_intervals_rpm_full = np.diff(peaks_rpm)/fs
                    window_size_rpm_avg_full = min(len(peak_intervals_rpm_full), config.RESP_OF_RPM_MOVING_AVG_INTERVALS)
                    if window_size_rpm_avg_full > 0:
                        moving_avg_period_rpm_full = np.convolve(peak_intervals_rpm_full, np.ones(window_size_rpm_avg_full)/window_size_rpm_avg_full, mode='valid').mean()
                        if moving_avg_period_rpm_full > eps:
                            avg_rpm_val = 60.0 / moving_avg_period_rpm_full
                            avg_rpm_str = f"Avg. Rate: {avg_rpm_val:.1f} breaths/min"
            plt.title(f"Filtered Respiration Signal - {avg_rpm_str}")
        else:
            plt.subplot(212); plt.plot(times_resp_full, med_filt_resp_full, color='m', label="Detrended & Median Filtered Resp (Bandpass Filter Error)"); plt.title(f"Processed Respiration Signal - {avg_rpm_str}")
            
        plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.legend(); plt.grid(True)
        plt.tight_layout(rect=[0,0,1,0.95]); plt.savefig(config.FINAL_RESPIRATION_PLOT_FILENAME); plt.show(block=True)