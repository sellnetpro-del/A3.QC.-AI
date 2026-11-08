
import os
import io
import logging

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from PIL import Image, UnidentifiedImageError

# ==========================================
# A3 QC AI - TEAM 48
# Punjab College Quaid Campus
# ICS CQ-31
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("A3-QC-AI")

app = FastAPI(
    title="A3 QC AI Platform",
    description="AI Platform developed by Team 48",
    version="2.1.0"
)

application = app

MODEL_NAME = "gemini-3.8-flash"

# ==========================================
# STATIC WEBSITE
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==========================================
# GEMINI API CLIENT
# ==========================================

def get_ai_client():

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing in Vercel Environment Variables."
        )

    return genai.Client(api_key=api_key)


# ==========================================
# HOME PAGE
# ==========================================

@app.get("/")
async def home():

    return FileResponse("static/index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/api/health")
async def health():

    return {
        "status": "online",
        "project": "A3 QC AI",
        "team": "Team 48",
        "model": MODEL_NAME
    }


# ==========================================
# GENERAL AI GENERATION
# ==========================================

@app.post("/api/generate")
async def generate_ai(
    module: str = Form(...),
    prompt_text: str = Form(...)
):

    if not prompt_text.strip():

        return JSONResponse(
            status_code=400,
            content={
                "detail": "Please enter some text."
            }
        )

    try:

        client = get_ai_client()

        system_context = """
You are A3 QC AI, an intelligent educational
and productivity assistant developed by Team 48
at Punjab College Quaid Campus, ICS CQ-31.

Provide helpful, accurate, clear and professional
answers.

For educational questions, explain concepts simply.
For career guidance, provide realistic pathways.
For budget questions, provide responsible general
budgeting suggestions.
For language tasks, follow the requested language
and formatting.

Use headings and bullet points when helpful.
"""

        full_prompt = f"""
{system_context}

MODULE: {module}

USER REQUEST:
{prompt_text}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt
        )

        result = response.text

        if not result:
            result = "The AI could not generate a response. Please try again."

        return {
            "success": True,
            "module": module,
            "result": result
        }

    except ValueError as e:

        logger.exception("Configuration error")

        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e)
            }
        )

    except Exception as e:

        logger.exception("AI generation failed")

        return JSONResponse(
            status_code=500,
            content={
                "detail": f"AI ERROR: {str(e)}"
            }
        )


# ==========================================
# AI WASTE CLASSIFIER
# Text + Image Support
# ==========================================

@app.post("/api/classify-waste")
async def classify_waste(
    description: str = Form(default=""),
    file: UploadFile = File(default=None)
):

    if not description.strip() and (
        file is None or not file.filename
    ):

        return JSONResponse(
            status_code=400,
            content={
                "detail": "Please enter a waste description or upload an image."
            }
        )

    try:

        client = get_ai_client()

        contents = [
            """
You are Eco Vision, the waste classification
assistant of A3 QC AI.

Identify the waste item and classify it into
one of these categories:

1. Plastic
2. Glass
3. Metal
4. Organic
5. E-waste
6. Paper
7. Other / Mixed Waste

Provide:

- Identified item
- Waste category
- Brief explanation
- Recommended recycling or disposal method
- Environmental benefit

If the image is unclear, explain that the
classification is uncertain.
"""
        ]

        # IMAGE PROCESSING

        if file is not None and file.filename:

            image_bytes = await file.read()

            if len(image_bytes) > 10 * 1024 * 1024:

                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": "Image size must be 10 MB or less."
                    }
                )

            try:

                image = Image.open(
                    io.BytesIO(image_bytes)
                )

                image.verify()

                image = Image.open(
                    io.BytesIO(image_bytes)
                ).convert("RGB")

                contents.append(image)

            except (
                UnidentifiedImageError,
                OSError,
                ValueError
            ):

                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": "Invalid image. Please upload a valid image file."
                    }
                )

        # TEXT DESCRIPTION

        if description.strip():

            contents.append(
                "Waste description: " + description.strip()
            )

        # GEMINI GENERATION

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents
        )

        result = response.text

        if not result:

            result = "Unable to classify this waste item. Please try again."

        return {
            "success": True,
            "result": result
        }

    except ValueError as e:

        logger.exception("Configuration error")

        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e)
            }
        )

    except Exception as e:

        logger.exception("Waste classification failed")

        return JSONResponse(
            status_code=500,
            content={
                "detail": f"WASTE AI ERROR: {str(e)}"
            }
        )


# ==========================================
# END OF A3 QC AI BACKEND
# ==========================================
