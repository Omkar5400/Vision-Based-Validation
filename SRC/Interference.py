import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

# --- 1. CONFIGURATION ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "airbag_unet_v2.pth"
VIDEO_PATH = "VehDB_14257_all_files/videos/Camera No. 8 - Passenger Close-Up.mp4"
T_ZERO_FRAME = 1 

# Physics & Metric Parameters
FPS = 1000 
PIXEL_TO_METER = 0.002 # 1 pixel = 2mm
OCCUPANT_HEAD_XY = (220, 350) # Static landmark for clearance demo

# --- 2. INITIALIZATION ---
model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

cap = cv2.VideoCapture(VIDEO_PATH)

# Analytics Buffers
mask_buffer = [] 
area_history = []
prev_gray = None
max_velocity = 0
ttp_ms = 0

print(f"--- BMW Safety Analytics Suite: Online ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    display_frame = frame.copy()
    
    if current_frame >= T_ZERO_FRAME:
        # --- PHASE 1 & 2: AI SEGMENTATION & FILTERING ---
        input_img = cv2.resize(frame, (256, 256))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            ai_mask = torch.sigmoid(output) > 0.9  
            ai_mask = ai_mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        full_ai_mask = cv2.resize(ai_mask, (frame.shape[1], frame.shape[0]))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_pixels = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        refined_mask = cv2.bitwise_and(full_ai_mask, bright_pixels)

        # Blob & Temporal Smoothing
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
        raw_mask = np.zeros_like(refined_mask)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            sorted_indices = np.argsort(areas)[::-1]
            for i in range(min(2, len(sorted_indices))):
                idx = sorted_indices[i] + 1
                if stats[idx, cv2.CC_STAT_AREA] > 150:
                    raw_mask[labels == idx] = 255
        
        mask_buffer.append(raw_mask)
        if len(mask_buffer) > 3: mask_buffer.pop(0)
        avg_mask = np.mean(mask_buffer, axis=0).astype(np.uint8)
        _, final_mask = cv2.threshold(avg_mask, 127, 255, cv2.THRESH_BINARY)

        # --- PHASE 3: METRIC DERIVATION ---
        
        # A. Velocity Mapping (Optical Flow)
        velocity_mps = 0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_masked = cv2.bitwise_and(mag, mag, mask=final_mask)
            velocity_mps = np.max(mag_masked) * PIXEL_TO_METER * FPS
        prev_gray = gray

        # B. Area Estimation & Time-to-Peak (TTP)
        airbag_area_px = np.count_nonzero(final_mask)
        airbag_area_m2 = airbag_area_px * (PIXEL_TO_METER ** 2)
        area_history.append(airbag_area_px)
        
        if airbag_area_px == max(area_history) and airbag_area_px > 500:
            ttp_ms = (current_frame - T_ZERO_FRAME) * (1000 / FPS)

        # C. Clearance Monitoring (Leading Edge to Occupant)
        clearance_mm = 0
        airbag_coords = np.where(final_mask > 0)
        if len(airbag_coords[1]) > 0:
            # Finding the point of the airbag closest to the occupant (left side of frame)
            leading_edge_x = np.min(airbag_coords[1])
            leading_edge_y = airbag_coords[0][np.argmin(airbag_coords[1])]
            
            dist_px = np.sqrt((leading_edge_x - OCCUPANT_HEAD_XY[0])**2 + (leading_edge_y - OCCUPANT_HEAD_XY[1])**2)
            clearance_mm = dist_px * PIXEL_TO_METER * 1000
            
            # Draw Clearance Line
            cv2.line(display_frame, (leading_edge_x, leading_edge_y), OCCUPANT_HEAD_XY, (255, 100, 0), 2)
            cv2.circle(display_frame, OCCUPANT_HEAD_XY, 5, (0, 0, 255), -1)

        # --- VISUALIZATION ---
        overlay = frame.copy()
        overlay[final_mask > 0] = [0, 255, 0] 
        display_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Dashboard UI (Phase 3 Outputs)
        ui_color = (0, 255, 0)
        cv2.putText(display_frame, f"VELOCITY: {velocity_mps:.2f} m/s", (40, 50), 2, 0.6, (0, 255, 255), 2)
        cv2.putText(display_frame, f"CLEARANCE: {clearance_mm:.1f} mm", (40, 80), 2, 0.6, (255, 150, 0), 2)
        cv2.putText(display_frame, f"SURFACE AREA: {airbag_area_m2:.4f} m2", (40, 110), 2, 0.6, (200, 200, 0), 2)
        cv2.putText(display_frame, f"TIME TO PEAK: {ttp_ms:.1f} ms", (40, 140), 2, 0.6, (0, 165, 255), 2)
    
    cv2.imshow("BMW Passive Safety Suite", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()