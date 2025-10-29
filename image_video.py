import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
import json
import uuid
import re
from groq import Groq
import shutil
import tempfile

# ---------------- Utility ----------------
def clean_plate_text(text: str) -> str:
    text = re.sub(r"The text.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"And.*?:", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^A-Z0-9 ]", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text.strip()

# ---------------- Load Models ----------------
# custom_model_path = r"C:\Users\AmanFarkade\OneDrive - Pepper India Resolution Private Limited\Aman\Aman\hel_det\project_one\trafic_violation_2\best.pt"
# custom_model = YOLO(custom_model_path)
custom_model = YOLO("best.pt")
coco_model = YOLO("yolo11n.pt")  # Pretrained YOLO

num_plate_dir = "num_plates"
os.makedirs(num_plate_dir, exist_ok=True)

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

def extract_text_from_image(image_path):
    try:
        with open(image_path, "rb") as f:
            b64_img = base64.b64encode(f.read()).decode("utf-8")

        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract only the vehicle number plate text in valid Indian format (e.g., MH12AB1234, MP09C5678, DL1CAB1234). If it does not match, respond with 'Unknown number'. If unreadable, respond with 'Unable to read'."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}" }},
                    ],
                }
            ]
            ,
            temperature=0,
            max_completion_tokens=512,
        )

        extracted_text = completion.choices[0].message.content
        return clean_plate_text(extracted_text)
    except:
        return "Unable to read"

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Helmet & Violation Detection", layout="wide")
st.title("🚦Traffic Violation Detection")

input_mode = st.radio("Choose Input Type", ["Image", "Video"])

# ---------------- IMAGE INPUT ----------------
if input_mode == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img_path = os.path.join("temp_input.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run YOLO detections
        custom_results = custom_model.predict(source=img_path, conf=0.5, save=False)
        coco_results = coco_model.predict(source=img_path, conf=0.5, save=False)

        orig_img = cv2.imread(img_path)
        annotated_img = orig_img.copy()

        # (Keep your detection + violation logic here exactly as before...)
            # ---------------- Detection Storage ----------------
        motorcycle_boxes = []
        helmet_boxes = []  # helmet + no-helmet
        other_boxes = []

        # ---------------- Draw Boxes ----------------
        def draw_boxes(results, names):
            # remove this line ↓
            # nonlocal annotated_img, motorcycle_boxes, helmet_boxes, other_boxes
            for r in results.boxes:
                cls_id = int(r.cls[0])
                label = names[cls_id]
                xyxy = r.xyxy[0].cpu().numpy().astype(int)
                conf = float(r.conf[0])

                # Store detections
                if label == "motorcycle":
                    motorcycle_boxes.append((xyxy, conf, label))
                elif label in ["helmet", "no-helmet"]:
                    helmet_boxes.append((xyxy, conf, label))
                else:
                    other_boxes.append((xyxy, conf, label))

                # Colors
                if label == "bicycle":
                    color = (0, 255, 255)
                elif label == "motorcycle":
                    color = (0, 0, 255)
                elif label == "helmet":
                    color = (0, 255, 0)
                elif label == "no-helmet":
                    color = (0, 165, 255)
                else:
                    color = (255, 0, 0)

                # Draw on annotated image
                cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(
                    annotated_img,
                    f"{label} {conf*100:.1f}",
                    (xyxy[0], xyxy[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        # Run both models
        draw_boxes(custom_results[0], custom_results[0].names)
        draw_boxes(coco_results[0], coco_results[0].names)

        # ---------------- Improved Violation Detection ----------------
        num_plate_dir = "num_plates"
        os.makedirs(num_plate_dir, exist_ok=True)

        def get_number_plate(motorcycle_box):
            for other_box, _, other_label in other_boxes:
                if "number" in other_label.lower() or "plate" in other_label.lower():
                    if (other_box[0] >= motorcycle_box[0] and
                        other_box[1] >= motorcycle_box[1] and
                        other_box[2] <= motorcycle_box[2] and
                        other_box[3] <= motorcycle_box[3]):

                        num_plate_img = orig_img[other_box[1]:other_box[3], other_box[0]:other_box[2]]
                        num_plate_path = os.path.join(num_plate_dir, f"numplate_{motorcycle_box[0]}.jpg")
                        cv2.imwrite(num_plate_path, num_plate_img)
                        if num_plate_img.size == 0:
                            print("⚠️ Empty crop detected for number plate!")
                        else:
                            print(f"✅ Cropped plate saved at: {num_plate_path}")
                        return extract_text_from_image(num_plate_path)
            return "Unknown"

        def is_head_above_motorcycle(helmet_box, motorcycle_box):
            helmet_center_x = (helmet_box[0] + helmet_box[2]) // 2
            helmet_bottom_y = helmet_box[3]
            motorcycle_center_x = (motorcycle_box[0] + motorcycle_box[2]) // 2
            motorcycle_top_y = motorcycle_box[1]

            x_distance = abs(helmet_center_x - motorcycle_center_x)
            y_distance = motorcycle_top_y - helmet_bottom_y

            return (y_distance > 0 and y_distance < 200 and 
                    x_distance < (motorcycle_box[2] - motorcycle_box[0]) * 0.4)

        def detect_violations():
            violations = []
            for motorcycle_box, motorcycle_conf, _ in motorcycle_boxes:
                number_plate = get_number_plate(motorcycle_box)
                riders = []

                for helmet_box, helmet_conf, helmet_label in helmet_boxes:
                    if is_head_above_motorcycle(helmet_box, motorcycle_box):
                        riders.append((helmet_box, helmet_label, helmet_conf))

                if riders:
                    riders_without_helmet = [r for r in riders if r[1] == "no-helmet"]
                    if riders_without_helmet:
                        violations.append(f"🚨 Rider without helmet | Plate: {number_plate}")
                    if len(riders) > 2:
                        violations.append(f"🚨 Triple seat violation ({len(riders)} riders) | Plate: {number_plate}")
                    # if len(riders_without_helmet) > 1:
                    #     violations.append(f"🚨 Multiple riders without helmets ({len(riders_without_helmet)}) | Plate: {number_plate}")
            return violations

        # ---------------- Run Violation Detection ----------------
        violations_found = detect_violations()

        # ---------------- Convert Violations to Dict ----------------
        violations_dict = {"violations": []}

        for v in violations_found:
            if "helmet" in v.lower():
                v_type = "Rider without helmet"
            elif "triple seat" in v.lower():
                v_type = "Triple seat violation"
            elif "multiple riders" in v.lower():
                v_type = "Multiple riders without helmets"
            else:
                v_type = "Other violation"

            # extract plate (after "| Plate: ...")
            if "| Plate:" in v:
                plate = v.split("| Plate:")[-1].strip()
            else:
                plate = "Unknown"

            violations_dict["violations"].append({
                "type": v_type,
                "plate": plate
            })

        # ---------------- Show Results ----------------
        st.subheader("📊 Detected Violations")
        if not violations_found:
            st.success("✅ No traffic violations detected.")
        else:
            for v in violations_found:
                st.error(v)

            unique = {tuple(sorted(v.items())) for v in violations_dict["violations"]}
            # Convert back to list of dicts
            violations_dict["violations"] = [dict(v) for v in unique]

            json_data = json.dumps(violations_dict, indent=4)

            # Download button
            st.download_button(
                label="📥 Download Violations JSON (.txt)",
                data=json_data,
                file_name="traffic_violations.txt",
                mime="text/plain"
            )

        st.subheader("📷 Processed Image")
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

elif input_mode == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        video_path = os.path.join("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(video_path)  # Show original video

        cap = cv2.VideoCapture(video_path)
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = temp_video.name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)

        # ---------------- Violations JSON Storage ----------------
        violations_log = {"video_violations": []}

        frame_num = 0
        frame_skip = 30  # analyze 1 in every 5 frames
        plate_cache = {}  # cache for OCR results
        num_plate_dir = "num_plates"
        os.makedirs(num_plate_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to save processing time
            if frame_num % frame_skip != 0:
                out.write(frame)   # still write frame into output video
                frame_num += 1
                continue

            # Run YOLO detections on frame
            results_custom = custom_model.predict(source=frame, conf=0.5, verbose=False)
            results_coco = coco_model.predict(source=frame, conf=0.5, verbose=False)

            annotated_frame = frame.copy()

            motorcycle_boxes = []
            helmet_boxes = []
            other_boxes = []

            # ---------------- Draw Boxes ----------------
            def draw_boxes(results, names):
                for r in results.boxes:
                    cls_id = int(r.cls[0])
                    label = names[cls_id]
                    xyxy = r.xyxy[0].cpu().numpy().astype(int)
                    conf = float(r.conf[0])

                    if label == "motorcycle":
                        motorcycle_boxes.append((xyxy, conf, label))
                    elif label in ["helmet", "no-helmet"]:
                        helmet_boxes.append((xyxy, conf, label))
                    else:
                        other_boxes.append((xyxy, conf, label))

                    # Pick color
                    if label == "bicycle":
                        color = (0, 255, 255)
                    elif label == "motorcycle":
                        color = (0, 0, 255)
                    elif label == "helmet":
                        color = (0, 255, 0)
                    elif label == "no-helmet":
                        color = (0, 165, 255)
                    else:
                        color = (255, 0, 0)

                    cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]),
                                (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(annotated_frame,
                                f"{label} {conf*100:.1f}",
                                (xyxy[0], xyxy[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

            draw_boxes(results_custom[0], results_custom[0].names)
            draw_boxes(results_coco[0], results_coco[0].names)

            # ---------------- Number Plate + Violation Detection ----------------
            def get_number_plate(motorcycle_box):
                for other_box, _, other_label in other_boxes:
                    if "number" in other_label.lower() or "plate" in other_label.lower():
                        if (other_box[0] >= motorcycle_box[0] and
                            other_box[1] >= motorcycle_box[1] and
                            other_box[2] <= motorcycle_box[2] and
                            other_box[3] <= motorcycle_box[3]):

                            num_plate_img = frame[other_box[1]:other_box[3],
                                                  other_box[0]:other_box[2]]
                            num_plate_path = os.path.join(
                                num_plate_dir, f"numplate_{uuid.uuid4().hex}.jpg"
                            )
                            cv2.imwrite(num_plate_path, num_plate_img)
                            if num_plate_img.size == 0:
                                return "Unknown"
                            return extract_text_from_image(num_plate_path)
                return "Unknown"

            def is_head_above_motorcycle(helmet_box, motorcycle_box):
                helmet_center_x = (helmet_box[0] + helmet_box[2]) // 2
                helmet_bottom_y = helmet_box[3]
                motorcycle_center_x = (motorcycle_box[0] + motorcycle_box[2]) // 2
                motorcycle_top_y = motorcycle_box[1]

                x_distance = abs(helmet_center_x - motorcycle_center_x)
                y_distance = motorcycle_top_y - helmet_bottom_y

                return (y_distance > 0 and y_distance < 200 and
                        x_distance < (motorcycle_box[2] - motorcycle_box[0]) * 0.4)

            def detect_violations():
                violations = []
                for motorcycle_box, motorcycle_conf, _ in motorcycle_boxes:
                    number_plate = get_number_plate(motorcycle_box)
                    riders = []

                    for helmet_box, helmet_conf, helmet_label in helmet_boxes:
                        if is_head_above_motorcycle(helmet_box, motorcycle_box):
                            riders.append((helmet_box, helmet_label, helmet_conf))

                    if riders:
                        riders_without_helmet = [r for r in riders if r[1] == "no-helmet"]
                        if riders_without_helmet:
                            violations.append({"type": "Rider without helmet",
                                               "plate": number_plate})
                        if len(riders) > 2:
                            violations.append({"type": f"Triple seat violation ({len(riders)} riders)",
                                               "plate": number_plate})
                        # if len(riders_without_helmet) > 1:
                        #     violations.append({"type": f"Multiple riders without helmets ({len(riders_without_helmet)})",
                        #                        "plate": number_plate})
                return violations

            detected = detect_violations()

            if detected:
                for v in detected:
                    v_entry = {
                        "frame": frame_num,
                        "type": v["type"],
                        "plate": v["plate"]
                    }
                    violations_log["video_violations"].append(v_entry)

                    # --- Directly show violation frame ---
                    st.markdown(f"**Frame {frame_num} | {v['type']} | Plate: {v['plate']}**")
                    st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

            # ---------------- Write Frame ----------------
            out.write(annotated_frame)

            frame_num += 1
            progress.progress(min(frame_num / frame_count, 1.0))

        cap.release()
        out.release()

        # st.success("✅ Video processing completed!")
        # st.video(out_path)

        def clean_plate_text(text: str) -> str:
            if not text or "unknown" in text.lower():
                return "Unknown"
            # Keep only alphanumeric + spaces
            cleaned = re.sub(r'[^A-Z0-9 ]', '', text.upper())
            # Remove multiple spaces
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned if cleaned else "Unknown"

        # ---------------- Save Violations JSON ----------------
        if violations_log["video_violations"]:
            seen = set()
            dedup = []
            for v in violations_log["video_violations"]:
                plate = clean_plate_text(v["plate"])
                v_type = v["type"]

                key = (plate, v_type)
                if key not in seen:
                    seen.add(key)
                    dedup.append({
                        "frame": v["frame"],   # keep the first frame occurrence
                        "type": v_type,
                        "plate": plate
                    })

            violations_log["video_violations"] = dedup

            # Convert to JSON
            json_data = json.dumps(violations_log, indent=4)

            st.subheader("📊 Violations JSON Log")
            st.json(violations_log)

            # Show violation frames
            for v in violations_log["video_violations"]:
                if "frame_image" in v and os.path.exists(v["frame_image"]):
                    st.markdown(f"**Frame {v['frame']} | {v['type']} | Plate: {v['plate']}**")
                    st.image(cv2.cvtColor(cv2.imread(v["frame_image"]), cv2.COLOR_BGR2RGB), use_container_width=True)

            st.download_button(
                label="📥 Download Video Violations JSON",
                data=json_data,
                file_name="video_violations.json",
                mime="application/json"
            )
            try:
                shutil.rmtree(num_plate_dir)  # deletes the whole temp folder
                os.makedirs(num_plate_dir, exist_ok=True)  # recreate empty
                print("✅ Temp number plate images deleted")
            except Exception as e:
                print(f"⚠️ Error cleaning number plate images: {e}")
        else:
            st.success("✅ No violations detected in the video!")
        st.success("✅ Video processing completed!")
        # with open(out_path, "rb") as f:
        #     st.video(f.read())


    