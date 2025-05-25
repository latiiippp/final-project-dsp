import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageTk

# Assuming main.py's logic is refactored into rppg_processor_module.py
# Create this file by moving the core processing from main.py into the RPPGProcessor class
try:
    from rppg_processor_module import RPPGProcessor
except ImportError:
    messagebox.showerror("Import Error", "Could not import RPPGProcessor from rppg_processor_module.py. Please ensure it's correctly refactored and in the same directory.")
    RPPGProcessor = None # Fallback

class RespirationMonitorApp:
    def __init__(self, window, window_title, camera_id=0):
        self.window = window
        self.window.title(window_title)

        self.camera_id = camera_id
        self.cap = None
        self.video_writer = None
        self.is_camera_running = False
        self.is_recording = False

        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 30
        self.current_bpm = 0.0
        
        # Initialize the RPPG Processor
        if RPPGProcessor:
            self.rppg_processor = RPPGProcessor()
        else:
            self.rppg_processor = None # Processor could not be imported

        # GUI Elements (same as before)
        self.main_frame = ttk.Frame(window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, columnspan=3, pady=5)
        self.btn_start_stop_camera = ttk.Button(self.main_frame, text="Start Camera", command=self.toggle_camera)
        self.btn_start_stop_camera.grid(row=1, column=0, padx=5, pady=5, sticky=tk.EW)
        self.btn_start_stop_record = ttk.Button(self.main_frame, text="Start Recording", command=self.toggle_recording, state=tk.DISABLED)
        self.btn_start_stop_record.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        self.bpm_label_text = tk.StringVar()
        self.bpm_label_text.set("BPM: N/A")
        self.bpm_label = ttk.Label(self.main_frame, textvariable=self.bpm_label_text, font=("Arial", 14))
        self.bpm_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.EW)
        self.status_bar_text = tk.StringVar()
        self.status_bar_text.set("Ready. Refactor main.py into rppg_processor_module.py if not done.")
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_bar_text, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.main_frame.columnconfigure(0, weight=1); self.main_frame.columnconfigure(1, weight=1); self.main_frame.columnconfigure(2, weight=1)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_interval = 30

    def initialize_camera_resources(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Could not open webcam (ID: {self.camera_id}).")
            return False

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.actual_fps == 0: self.actual_fps = 30
        
        if self.rppg_processor:
            # Pass necessary parameters to the processor
            # The processor's __init__ might take these, or have a separate setup method
            pass # self.rppg_processor.some_setup_method(self.actual_fps, ...)
        else:
            messagebox.showerror("Error", "RPPG Processor not available.")
            return False
            
        self.status_bar_text.set(f"Camera: {self.actual_width}x{self.actual_height} @ {self.actual_fps:.0f} FPS")
        return True

    def toggle_camera(self):
        if not self.rppg_processor:
            messagebox.showerror("Error", "RPPG Processor module not loaded. Cannot start camera.")
            return

        if self.is_camera_running:
            self.is_camera_running = False
            self.btn_start_stop_camera.config(text="Start Camera")
            self.btn_start_stop_record.config(state=tk.DISABLED)
            if self.is_recording: self.toggle_recording()
            if self.cap: self.cap.release(); self.cap = None
            # self.rppg_processor.close() # Processor might have its own close for MediaPipe
            self.video_label.config(image=''); self.status_bar_text.set("Camera stopped.")
            self.bpm_label_text.set("BPM: N/A")
        else:
            if self.initialize_camera_resources():
                self.is_camera_running = True
                self.btn_start_stop_camera.config(text="Stop Camera")
                self.btn_start_stop_record.config(state=tk.NORMAL)
                # Tell processor to start collecting data
                self.rppg_processor.set_capture_parameters(self.actual_fps, time.time())
                self.update_frame()

    def toggle_recording(self):
        # This logic remains largely the same, as it's GUI-specific
        if self.is_recording:
            self.is_recording = False
            self.btn_start_stop_record.config(text="Start Recording")
            if self.video_writer: self.video_writer.release(); self.video_writer = None
            self.status_bar_text.set("Recording stopped.")
        else:
            if not self.is_camera_running or not self.cap: return
            filename = f"rppg_gui_output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if self.actual_width > 0 and self.actual_height > 0 and self.actual_fps > 0:
                self.video_writer = cv2.VideoWriter(filename, fourcc, float(self.actual_fps), (self.actual_width, self.actual_height))
                self.is_recording = True; self.btn_start_stop_record.config(text="Stop Recording")
                self.status_bar_text.set(f"Recording to {filename}")
            else:
                messagebox.showerror("Video Error", "Invalid frame dimensions/FPS for recording.")

    def update_frame(self):
        if not self.is_camera_running or not self.cap or not self.cap.isOpened() or not self.rppg_processor:
            return

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            self.status_bar_text.set("Error: Failed to grab frame.")
            if self.is_camera_running: self.window.after(self.update_interval, self.update_frame)
            return

        frame_bgr = cv2.flip(frame_bgr, 1)
        
        # --- Delegate processing to RPPGProcessor ---
        # The processor handles all annotations, signal extraction, BPM, and mini-graph generation
        annotated_frame, self.current_bpm, raw_graph_img, filtered_graph_img = \
            self.rppg_processor.process_single_frame(frame_bgr, self.actual_width, self.actual_height)
        # --- End delegation ---

        self.bpm_label_text.set(f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: N/A")

        # --- Overlay mini-graphs from processor onto the annotated_frame ---
        # (This assumes rppg_processor.process_single_frame returns the small graph images)
        # Example:
        graph_h, graph_w = raw_graph_img.shape[:2]
        y_offset_raw = 10
        x_offset_graphs = self.actual_width - graph_w - 10
        
        if y_offset_raw + graph_h < self.actual_height and x_offset_graphs > 0:
            annotated_frame[y_offset_raw : y_offset_raw + graph_h, x_offset_graphs : x_offset_graphs + graph_w] = raw_graph_img
            cv2.putText(annotated_frame, "Raw G", (x_offset_graphs, y_offset_raw - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

        y_offset_filt = y_offset_raw + graph_h + 10
        if y_offset_filt + graph_h < self.actual_height and x_offset_graphs > 0:
            annotated_frame[y_offset_filt : y_offset_filt + graph_h, x_offset_graphs : x_offset_graphs + graph_w] = filtered_graph_img
            cv2.putText(annotated_frame, "Filtered G", (x_offset_graphs, y_offset_filt - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
            bpm_text_video = f"BPM: {self.current_bpm:.1f}" if self.current_bpm > 0 else "BPM: N/A"
            cv2.putText(annotated_frame, bpm_text_video, (x_offset_graphs, y_offset_filt + graph_h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)
        # --- End graph overlay ---

        # Display in Tkinter
        img_rgb_tk = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb_tk)
        max_disp_w = 800
        if self.actual_width > max_disp_w:
            ratio = max_disp_w / self.actual_width
            img_pil = img_pil.resize((max_disp_w, int(self.actual_height * ratio)), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = img_tk
        self.video_label.config(image=img_tk)

        if self.is_recording and self.video_writer:
            self.video_writer.write(annotated_frame) # Write frame with overlays

        if self.is_camera_running:
            self.window.after(self.update_interval, self.update_frame)

    def generate_final_plot(self):
        if not self.rppg_processor: return
        
        collected_signal, collected_time = self.rppg_processor.get_collected_data_for_plotting()

        if not collected_signal or len(collected_signal) < self.actual_fps * 2: # Min 2s data
            print("GUI: Not enough data from processor for final plot.")
            return
        
        print("GUI: Generating final Matplotlib plot from processor data...")
        # (Plotting logic remains similar, but uses data from rppg_processor)
        # You'll need to ensure scipy.signal is imported if filtering is done here for the plot
        # or if the processor provides already filtered data for plotting.
        # For simplicity, assuming processor provides raw, and we filter here for plot:
        
        final_raw = np.array(collected_signal)
        final_time = np.array(collected_time)
        min_len = min(len(final_raw), len(final_time))
        final_raw, final_time = final_raw[:min_len], final_time[:min_len]

        if len(final_raw) <= 2 * self.rppg_processor.filter_order + 1: return

        from scipy import signal # Import for plotting if needed here
        detrended = signal.detrend(final_raw)
        filtered_plot_sig = None
        fs_plot = self.rppg_processor.fs
        lc_plot, hc_plot = self.rppg_processor.lowcut, self.rppg_processor.highcut
        fo_plot = self.rppg_processor.filter_order

        if fs_plot > 0 and (hc_plot * 2 < fs_plot):
            nyq,low,high = 0.5*fs_plot, lc_plot/(0.5*fs_plot), hc_plot/(0.5*fs_plot)
            if low>0 and high<1 and low<high:
                b,a = signal.butter(fo_plot, [low,high], btype='band')
                filtered_plot_sig = signal.filtfilt(b,a,detrended)
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1); plt.plot(final_time, final_raw, label='Raw rPPG (Processor)'); plt.title('Raw rPPG'); plt.legend(); plt.grid(True)
        plt.subplot(2,1,2)
        if filtered_plot_sig is not None:
            plt.plot(final_time, filtered_plot_sig, label='Filtered rPPG (Processor)', color='orange')
        else:
            plt.plot(final_time, detrended, label='Detrended rPPG (Filter Failed)', color='red')
        plt.title('Processed rPPG'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plot_fname = f"rppg_gui_final_plot_proc_{time.strftime('%Y%m%d_%H%M%S')}.png"
        try: plt.savefig(plot_fname); print(f"GUI: Plot saved to {plot_fname}")
        except Exception as e: print(f"GUI: Error saving plot: {e}")


    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            if self.is_camera_running: self.is_camera_running = False
            if self.cap: self.cap.release()
            if self.video_writer: self.video_writer.release()
            if self.rppg_processor:
                self.generate_final_plot() # Generate plot with data from processor
                self.rppg_processor.close() # Release processor's resources
            self.window.destroy()

if __name__ == '__main__':
    if RPPGProcessor is None:
        print("Exiting: RPPGProcessor class could not be loaded.")
    else:
        root = tk.Tk()
        app = RespirationMonitorApp(root, "Respiration Monitor GUI (using Processor)")
        root.mainloop()