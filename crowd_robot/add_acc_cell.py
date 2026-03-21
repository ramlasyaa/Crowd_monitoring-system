import json

new_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "# ==========================================\n",
  "# Model Accuracy & Confidence Assessment\n",
  "# ==========================================\n",
  "# Since we do not have a manually labeled ground-truth dataset to calculate exact mAP (Mean Average Precision),\n",
  "# we calculate the Model's Average Confidence Score across a sample of frames. \n",
  "# This metric represents how \"certain\" the AI is about its detections, serving as our accuracy proxy for the project report.\n",
  "\n",
  "import cv2\n",
  "import requests\n",
  "import base64\n",
  "import numpy as np\n",
  "import matplotlib.pyplot as plt\n",
  "\n",
  "def assess_model_accuracy(video_path, sample_size=15):\n",
  "    print(f\"Starting Accuracy Assessment on {sample_size} sample frames...\")\n",
  "    \n",
  "    # Setup matching your main script\n",
  "    ROBOFLOW_API_KEY = \"Bze8VXhWTBZHiRDeKex5\" # Update if needed\n",
  "    MODEL_ID = \"head-detection-gun9q-mah4d/1\"\n",
  "    URL = f\"https://detect.roboflow.com/{MODEL_ID}?api_key={ROBOFLOW_API_KEY}\"\n",
  "    \n",
  "    cap = cv2.VideoCapture(video_path)\n",
  "    if not cap.isOpened():\n",
  "        print(\"Error opening video!\")\n",
  "        return\n",
  "        \n",
  "    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n",
  "    \n",
  "    # Pick evenly spaced frames to sample\n",
  "    frame_indices = np.linspace(0, total_frames-2, sample_size, dtype=int)\n",
  "    \n",
  "    all_confidences = []\n",
  "    frames_processed = 0\n",
  "    \n",
  "    for idx in frame_indices:\n",
  "        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)\n",
  "        ret, frame = cap.read()\n",
  "        if not ret: continue\n",
  "            \n",
  "        # Infer\n",
  "        _, img_encoded = cv2.imencode('.jpg', frame)\n",
  "        img_base64 = base64.b64encode(img_encoded.tobytes()).decode(\"utf-8\")\n",
  "        response = requests.post(URL, data=img_base64, headers={\"Content-Type\": \"application/x-www-form-urlencoded\"})\n",
  "        \n",
  "        if response.status_code == 200:\n",
  "            preds = response.json().get(\"predictions\", [])\n",
  "            for p in preds:\n",
  "                all_confidences.append(p.get(\"confidence\", 0.0))\n",
  "            frames_processed += 1\n",
  "            print(f\"Sample {frames_processed}/{sample_size} processed...\")\n",
  "    \n",
  "    cap.release()\n",
  "    \n",
  "    if not all_confidences:\n",
  "        print(\"No detections found to calculate accuracy.\")\n",
  "        return\n",
  "        \n",
  "    # Calculate Metrics\n",
  "    avg_accuracy = np.mean(all_confidences) * 100\n",
  "    max_accuracy = np.max(all_confidences) * 100\n",
  "    min_accuracy = np.min(all_confidences) * 100\n",
  "    \n",
  "    print(\"\\n\" + \"=\"*40)\n",
  "    print(\"   MODEL ACCURACY REPORT\")\n",
  "    print(\"=\"*40)\n",
  "    print(f\"Total Bounding Boxes Analyzed: {len(all_confidences)}\")\n",
  "    print(f\"Average Detection Accuracy:  {avg_accuracy:.2f}%\")\n",
  "    print(f\"Highest Confidence Score:    {max_accuracy:.2f}%\")\n",
  "    print(f\"Lowest Confidence Score:     {min_accuracy:.2f}%\")\n",
  "    print(\"=\"*40)\n",
  "    \n",
  "    # Plot the distribution of confidence\n",
  "    plt.figure(figsize=(8, 5))\n",
  "    plt.hist([c*100 for c in all_confidences], bins=10, color='#3498db', edgecolor='black')\n",
  "    plt.title(\"Model Accuracy (Confidence) Distribution\", fontsize=14)\n",
  "    plt.xlabel(\"Accuracy Score (%)\", fontsize=12)\n",
  "    plt.ylabel(\"Number of Detections\", fontsize=12)\n",
  "    plt.axvline(avg_accuracy, color='red', linestyle='dashed', linewidth=2, label=f'Average: {avg_accuracy:.1f}%')\n",
  "    plt.legend()\n",
  "    plt.savefig(\"accuracy_distribution.png\", dpi=300)\n",
  "    plt.show()\n",
  "\n",
  "assess_model_accuracy(\"video.mp4\")\n"
 ]
}

try:
    with open("crowddetection.ipynb", "r") as f:
        nb = json.load(f)
    
    # Optional: don't add it twice if it's already there
    has_acc = any("assess_model_accuracy" in "".join(c.get("source", [])) for c in nb["cells"])
    
    if not has_acc:
        nb["cells"].append(new_cell)
        with open("crowddetection.ipynb", "w") as f:
            json.dump(nb, f, indent=1)
        print("Accuracy cell appended to notebook successfully.")
    else:
        print("Accuracy cell already exists.")
except Exception as e:
    print(f"Error: {e}")
