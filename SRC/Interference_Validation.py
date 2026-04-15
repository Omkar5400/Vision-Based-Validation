import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from scipy.stats import pearsonr

# --- 1. CONFIGURATION ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "airbag_unet_v2.pth"
VIDEO_PATH = "VehDB_14257_all_files/videos/Camera No. 8 - Passenger Close-Up.mp4"
T_ZERO_FRAME = 1 

# Physics Parameters
FPS = 1000 
PIXEL_TO_METER = 0.002 

# --- 2. INITIALIZATION ---
model = smp.Unet(encoder_name="resnet18", in_channels=3, classes=1)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

cap = cv2.VideoCapture(VIDEO_PATH)

# Phase 4 Buffers
ai_area_history = []
sim_area_history = []
mask_buffer = [] 
prev_gray = None

print(f"--- BMW Passive Safety Suite: Phase 4 Validation Mode ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    display_frame = frame.copy()
    
    if current_frame >= T_ZERO_FRAME:
        # --- AI SEGMENTATION & PRECISION FILTERING ---
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

        # Blob & Smoothing
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_mask)
        raw_mask = np.zeros_like(refined_mask)
        if num_labels > 1:
            idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            raw_mask[labels == idx] = 255
        
        mask_buffer.append(raw_mask)
        if len(mask_buffer) > 3: mask_buffer.pop(0)
        avg_mask = np.mean(mask_buffer, axis=0).astype(np.uint8)
        _, final_mask = cv2.threshold(avg_mask, 127, 255, cv2.THRESH_BINARY)

        # --- PHASE 3: METRIC EXTRACTION ---
        airbag_area_px = np.count_nonzero(final_mask)
        ai_area_history.append(airbag_area_px)

        # --- PHASE 4: LS-DYNA CORRELATION (SIMULATED) ---
        # Generate the 'Ground Truth' Nodal Displacement for this frame
        t = current_frame - T_ZERO_FRAME
        # Sigmoid curve representing ideal LS-DYNA volume expansion
        sim_val = 1 / (1 + np.exp(-0.12 * (t - 65))) 
        # Scale to match the pixel magnitude of the video
        sim_area_history.append(sim_val * 45000) 

        # Calculate Pearson Correlation every 10 frames
        correlation_str = "CORRELATION: Calculating..."
        if len(ai_area_history) > 10:
            r, _ = pearsonr(ai_area_history, sim_area_history)
            correlation_str = f"CORRELATION: {r*100:.2f}%"

        # --- VISUALIZATION ---
        overlay = frame.copy()
        overlay[final_mask > 0] = [0, 255, 0] 
        display_frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Dashboard UI
        cv2.putText(display_frame, f"AI AREA: {airbag_area_px} px", (40, 50), 2, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"SIM AREA: {int(sim_area_history[-1])} px", (40, 80), 2, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, correlation_str, (40, 110), 2, 0.7, (0, 165, 255), 2)
        cv2.putText(display_frame, "PHASE 4: VALIDATION ACTIVE", (40, 140), 2, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("BMW AI Passive Safety Validation Suite", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()