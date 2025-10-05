# sanity.py — minimal Gemini 1.5 check
# - Verifies API key is present
# - Lists models that support generateContent
# - Picks a valid model (prefers gemini-1.5-flash), then does a tiny multimodal JSON call

import os, io, json
from PIL import Image, ImageDraw
import google.generativeai as genai

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise SystemExit("Missing GOOGLE_API_KEY env var. Set it and rerun (see README).")

genai.configure(api_key=API_KEY)

# 1) List models that support generateContent
models = [m.name for m in genai.list_models()
          if "generateContent" in getattr(m, "supported_generation_methods", [])]
print("Supported models:", models)

# 2) Pick a good one
model_name = next((m for m in models if "models/gemini-1.5-flash" in m),
              next((m for m in models if "models/gemini-1.5-pro" in m), None))
if not model_name:
    raise SystemExit("No Gemini model with generateContent available in this account.")

print("Using:", model_name)

# 3) Build a tiny in-memory image
img = Image.new("RGB", (420, 140), "white")
d = ImageDraw.Draw(img)
d.text((10, 50), "Test receipt total $12.34", fill="black")

buf = io.BytesIO()
img.save(buf, format="PNG")
image_part = {"mime_type": "image/png", "data": buf.getvalue()}

prompt = 'Return JSON: {"ok": true, "note": "hello"}'

# 4) Call the model for JSON
model = genai.GenerativeModel(model_name)
resp = model.generate_content(
    [image_part, {"text": prompt}],
    generation_config={"response_mime_type": "application/json", "temperature": 0.2},
)

text = getattr(resp, "text", None)
if not text:
    # fallback parsing for some SDK shapes
    text = resp.candidates[0].content.parts[0].text

print("Raw response text:", text)
parsed = json.loads(text)  # should be valid JSON
print("Parsed JSON:", parsed)
print("✓ sanity.py succeeded.")
