import cv2
import numpy as np
import config
import utils 

def setup_display_and_video_writer(actual_width, actual_height, fs):
    display_config = {}
    display_config['panel_padding'] = config.DISPLAY_PANEL_PADDING
    display_config['graph_title_space'] = config.DISPLAY_GRAPH_TITLE_SPACE
    display_config['graph_text_color'] = config.DISPLAY_GRAPH_TEXT_COLOR
    display_config['grid_color'] = config.DISPLAY_GRID_COLOR
    display_config['graph_height'] = config.DISPLAY_GRAPH_HEIGHT # Ketinggian setiap plot individu

    # Tentukan TOTAL_WINDOW_WIDTH terutama dari config
    # Ini akan menjadi lebar utama window Anda.
    display_config['TOTAL_WINDOW_WIDTH'] = config.DISPLAY_MIN_TOTAL_WIDTH 

    # Hitung porsi untuk kamera dan grafik berdasarkan TOTAL_WINDOW_WIDTH
    display_config['display_camera_portion_w'] = int(display_config['TOTAL_WINDOW_WIDTH'] * config.DISPLAY_CAMERA_PORTION_W_FACTOR)
    display_config['display_graphs_portion_w'] = display_config['TOTAL_WINDOW_WIDTH'] - display_config['display_camera_portion_w']

    # Hitung tinggi tampilan kamera dengan menjaga rasio aspek
    camera_aspect_ratio = actual_height / actual_width if actual_width > 0 else 1.0
    display_config['display_camera_h'] = int(display_config['display_camera_portion_w'] * camera_aspect_ratio)
    
    # Jika display_camera_h menjadi 0 karena display_camera_portion_w kecil,
    # dan actual_height ada, coba berikan fallback minimal berdasarkan default cam height jika perlu,
    # atau biarkan 0 jika memang porsinya 0.
    if display_config['display_camera_portion_w'] > 0 and display_config['display_camera_h'] == 0 and actual_height > 0:
        # Ini bisa terjadi jika display_camera_portion_w sangat kecil.
        # Untuk mencegah tinggi kamera 0 jika lebarnya > 0 dan ada sumber tinggi.
        # Mungkin lebih baik membiarkannya 0 jika faktornya sangat kecil.
        # Untuk sekarang, jika lebarnya ada, tingginya juga harus ada jika sumbernya ada.
        default_aspect_ratio = config.DEFAULT_CAM_HEIGHT / config.DEFAULT_CAM_WIDTH if config.DEFAULT_CAM_WIDTH > 0 else 1.0
        min_sensible_cam_h = int(display_config['display_camera_portion_w'] * default_aspect_ratio)
        if min_sensible_cam_h > 0 : display_config['display_camera_h'] = min_sensible_cam_h
        # Jika masih 0, berarti display_camera_portion_w sangat kecil atau 0.

    # Lebar area untuk menggambar grafik individual
    display_config['graph_width'] = display_config['display_graphs_portion_w'] - (2 * display_config['panel_padding'])
    if display_config['graph_width'] < config.DISPLAY_MIN_GRAPH_WIDTH:
        # print(f"Warning: Calculated graph_width ({display_config['graph_width']}) is less than DISPLAY_MIN_GRAPH_WIDTH ({config.DISPLAY_MIN_GRAPH_WIDTH}). Adjusting.")
        display_config['graph_width'] = config.DISPLAY_MIN_GRAPH_WIDTH
        # Jika graph_width diubah, kita mungkin perlu menyesuaikan ulang display_graphs_portion_w dan TOTAL_WINDOW_WIDTH
        # Untuk saat ini, kita hanya memastikan graph_width tidak terlalu kecil.
        # Ini bisa berarti total window mungkin perlu lebih lebar dari yang di-target jika faktor kamera terlalu besar.
        # Atau, panel grafik akan mengambil ruang lebih banyak, mengurangi ruang kamera jika TOTAL_WINDOW_WIDTH tetap.
        # Untuk simplisitas, kita biarkan TOTAL_WINDOW_WIDTH seperti yang di-target, graph_width akan dibatasi minimum.

    # Hitung total tinggi yang dibutuhkan untuk semua grafik dan paddingnya
    graphs_area_h = (display_config['graph_height'] + display_config['graph_title_space'] + 15) * 4 + \
                    display_config['panel_padding'] * 1 + config.DISPLAY_LAYOUT_INFO_AREA_HEIGHT 
                    # Mengurangi jumlah panel_padding vertikal karena teks nilai di bawah grafik
    display_config['graphs_area_height_calc'] = graphs_area_h
    
    # Tinggi window gabungan adalah maksimum dari tinggi tampilan kamera atau tinggi area grafik
    display_config['combined_height'] = max(display_config['display_camera_h'], graphs_area_h)
    
    # Fallback jika kedua komponen tinggi adalah 0
    if display_config['display_camera_h'] == 0 and graphs_area_h == 0:
        display_config['combined_height'] = 480 # Default fallback height
    elif display_config['display_camera_h'] == 0 :
        display_config['combined_height'] = graphs_area_h
    elif graphs_area_h == 0: # Meskipun ini seharusnya tidak terjadi dengan 4 grafik
        display_config['combined_height'] = display_config['display_camera_h']


    display_config['combined_width'] = display_config['TOTAL_WINDOW_WIDTH']
    
    display_config['window_name'] = 'rPPG (POS) & Respiration (Optical Flow) - Modularized'
    cv2.namedWindow(display_config['window_name']) # Bisa ditambahkan WINDOW_NORMAL agar bisa di-resize manual
    # cv2.namedWindow(display_config['window_name'], cv2.WINDOW_NORMAL)


    gh, gw = display_config['graph_height'], display_config['graph_width']
    display_config['rppg_raw_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    display_config['rppg_filt_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    display_config['resp_raw_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    display_config['resp_filt_g'] = np.ones((gh,gw,3),dtype=np.uint8)*255
    
    # Pastikan video writer menggunakan combined_width dan combined_height yang sudah final
    if display_config['combined_width'] > 0 and display_config['combined_height'] > 0:
        display_config['video_writer'] = cv2.VideoWriter(config.OUTPUT_VIDEO_FILENAME, 
                                                         cv2.VideoWriter_fourcc(*'mp4v'), 
                                                         float(fs), 
                                                         (display_config['combined_width'], display_config['combined_height']))
    else:
        print("Warning: combined_width or combined_height is 0. Video writer not initialized.")
        display_config['video_writer'] = None

    display_config['r_panel_x'] = display_config['display_camera_portion_w']
    return display_config

def assemble_and_show_display(combined_display_bg_color, resized_frame_display, display_config, current_hr, current_rpm):
    # Gunakan np.full untuk menghindari masalah tipe data saat dikalikan
    combined_display_bg = np.full((display_config['combined_height'], display_config['combined_width'], 3), 
                                  combined_display_bg_color, dtype=np.uint8)
    
    # Penempatan frame kamera
    cam_h_to_place = min(display_config['display_camera_h'], combined_display_bg.shape[0])
    cam_w_to_place = min(display_config['display_camera_portion_w'], combined_display_bg.shape[1])

    if cam_h_to_place > 0 and cam_w_to_place > 0 and resized_frame_display is not None:
        # Pastikan resized_frame_display memiliki ukuran yang benar sebelum ditempatkan
        if resized_frame_display.shape[0] != cam_h_to_place or resized_frame_display.shape[1] != cam_w_to_place:
            resized_frame_display_safe = cv2.resize(resized_frame_display, (cam_w_to_place, cam_h_to_place))
        else:
            resized_frame_display_safe = resized_frame_display
        
        # Hitung offset y untuk kamera agar berada di tengah jika ada sisa ruang vertikal
        cam_y_offset = (display_config['combined_height'] - cam_h_to_place) // 2
        if cam_y_offset < 0: cam_y_offset = 0 # Pastikan tidak negatif

        combined_display_bg[cam_y_offset : cam_y_offset + cam_h_to_place, 0 : cam_w_to_place] = resized_frame_display_safe

    # Penempatan panel grafik
    # Hitung offset y untuk panel grafik agar berada di tengah jika ada sisa ruang vertikal
    graphs_total_h_needed = display_config['graphs_area_height_calc']
    graphs_y_offset = (display_config['combined_height'] - graphs_total_h_needed) // 2
    if graphs_y_offset < 0: graphs_y_offset = 0 # Pastikan tidak negatif
    
    current_graph_y = graphs_y_offset + display_config['panel_padding']


    graphs_data = [
        (display_config['rppg_raw_g'], "Raw POS rPPG", None, 0),
        (display_config['rppg_filt_g'], "Filtered POS (HR)", f"HR: {current_hr:.1f} BPM" if current_hr > 0 else "HR: N/A", current_hr),
        (display_config['resp_raw_g'], "Raw Respiration (OF Y-avg)", None, 0),
        (display_config['resp_filt_g'], "Filtered Respiration (OF)", f"RPM: {current_rpm:.1f}" if current_rpm > 0 else "RPM: N/A", current_rpm)
    ]

    graph_x_start = display_config['r_panel_x'] + display_config['panel_padding']
    
    for img_to_place, title_str, val_text_str, val_num in graphs_data:
        graph_h_current = display_config['graph_height']
        graph_w_current = display_config['graph_width']

        if current_graph_y + graph_h_current <= combined_display_bg.shape[0] and \
           graph_x_start + graph_w_current <= combined_display_bg.shape[1] and \
           img_to_place.shape[0] == graph_h_current and \
           img_to_place.shape[1] == graph_w_current:
            
            combined_display_bg[current_graph_y : current_graph_y + graph_h_current, 
                                graph_x_start : graph_x_start + graph_w_current] = img_to_place
            
            cv2.putText(combined_display_bg,title_str,(graph_x_start, current_graph_y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,display_config['graph_text_color'],1)
            if val_text_str: 
                cv2.putText(combined_display_bg,val_text_str,(graph_x_start + 5, current_graph_y + graph_h_current + 12),cv2.FONT_HERSHEY_SIMPLEX,0.45,display_config['graph_text_color'] if val_num > 0 else (100,100,100),1)
        
        current_graph_y += graph_h_current + display_config['graph_title_space'] + (15 if val_text_str else 0) # +15 untuk ruang teks nilai
            
    cv2.imshow(display_config['window_name'], combined_display_bg)
    if display_config.get('video_writer') is not None: 
        display_config['video_writer'].write(combined_display_bg)

def clear_and_draw_graph_grids(display_config):
    for g_key in ['rppg_raw_g', 'rppg_filt_g', 'resp_raw_g', 'resp_filt_g']:
        if display_config[g_key].shape[0] > 0 and display_config[g_key].shape[1] > 0: # Cek jika grafik valid
            display_config[g_key].fill(255)
            utils.draw_grid(display_config[g_key], display_config['graph_width'], display_config['graph_height'], display_config['grid_color'])