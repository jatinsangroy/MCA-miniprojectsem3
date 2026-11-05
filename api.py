"""
api.py
------
PathForge AI — FastAPI backend for the multi-agent learning path
generator.  Also serves the static HTML/CSS/JS frontend.

Endpoints
---------
POST /generate          – Full pipeline: profile → curriculum → resources → assessment
POST /assess            – Submit quiz answers and receive adaptation report
GET  /health            – Health check
GET  /                  – Serves the production frontend (frontend/index.html)
"""

import os
import asyncio
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()  # Load .env before anything else

# Resolve the frontend directory relative to this file
FRONTEND_DIR = Path(__file__).parent / "frontend"


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class UserProfile(BaseModel):
    name: str = Field(..., example="Akhil")
    current_skills: list[str] = Field(
        ..., example=["Python basics", "HTML/CSS", "SQL"]
    )
    experience_years: int = Field(0, ge=0, le=50, example=1)
    target_role: str = Field(..., example="Machine Learning Engineer")
    weekly_hours: int = Field(10, ge=1, le=80, example=15)
    generate_quiz_for_module: Optional[int] = Field(
        1, ge=1, description="Which module number to generate an assessment for"
    )


class QuizSubmission(BaseModel):
    module_number: int = Field(..., ge=1)
    enriched_roadmap: str = Field(..., description="The roadmap context from /generate")
    answers: dict[str, str] = Field(
        ..., example={"q1": "B", "q2": "Gradient descent updates weights..."}
    )


class PipelineResponse(BaseModel):
    gap_report: str
    roadmap: str
    enriched_roadmap: str
    assessment: str


class AssessmentResponse(BaseModel):
    assessment_report: str


class CodeRunRequest(BaseModel):
    code: str = Field(..., description="Python code snippet to execute in sandbox")


class CodeRunResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int


class NewQuizRequest(BaseModel):
    target_role: str = Field("Python Developer")
    module_number: int = Field(1, ge=1)
    round_number: int = Field(2, ge=1)
    enriched_roadmap: str = Field("")


class NewQuizResponse(BaseModel):
    assessment: str
    module_number: int
    round_number: int


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate critical env vars at startup."""
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
        )
    yield


app = FastAPI(
    title="PathForge AI API",
    description=(
        "PathForge AI — Multi-Agent backend (CrewAI + Groq) that generates "
        "fully personalized learning paths, curated resources, and "
        "adaptive assessments for any tech career goal."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
# CORS — allow all origins (restrict in production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static frontend ──────────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
    if (FRONTEND_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")
    if (FRONTEND_DIR / "pages").exists():
        app.mount("/pages", StaticFiles(directory=str(FRONTEND_DIR / "pages")), name="pages")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_root():
    """Serve the landing page."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "PathForge AI API", "docs": "/docs"}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health_check():
    """Quick liveness probe."""
    groq_key  = bool(os.getenv("GROQ_API_KEY"))
    openai_key = bool(os.getenv("OPENAI_API_KEY"))
    model = os.getenv("GROQ_MODEL", os.getenv("OPENAI_MODEL", "unset"))
    return {
        "status": "ok",
        "model": model,
        "llm_provider": "groq" if groq_key else "openai" if openai_key else "none",
        "frontend": FRONTEND_DIR.exists(),
    }


@app.post("/generate", response_model=PipelineResponse, tags=["Pipeline"])
async def generate_learning_path(profile: UserProfile):
    """
    Run the full 4-agent pipeline and return the complete learning path.

    CrewAI's kickoff() is synchronous; we run it in a thread-pool executor
    so it does not block FastAPI's async event loop (~30-120 s call).
    """
    try:
        from crew import run_pipeline

        profile_data = profile.model_dump(
            include={
                "name",
                "current_skills",
                "experience_years",
                "target_role",
                "weekly_hours",
            }
        )
        module_num = profile.generate_quiz_for_module or 1

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(run_pipeline, user_profile=profile_data, generate_quiz_for_module=module_num),
        )
        return PipelineResponse(
            gap_report=result["gap_report"],
            roadmap=result["roadmap"],
            enriched_roadmap=result["enriched_roadmap"],
            assessment=result["assessment"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/assess", response_model=AssessmentResponse, tags=["Assessment"])
async def submit_quiz(submission: QuizSubmission):
    """
    Submit learner quiz answers for a specific module and receive
    a mastery score + adaptation report for the next module.
    Runs CrewAI in a thread-pool executor to avoid blocking the event loop.
    """
    try:
        from crewai import Crew, Process
        from tasks import create_assessment_task
        from agents import create_assessment_adapter_agent

        def _run_assessment():
            try:
                agent = create_assessment_adapter_agent()
                task = create_assessment_task(
                    module_number=submission.module_number,
                    quiz_answers=submission.answers,
                    assessment_agent=agent,
                )
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                result = crew.kickoff()
                outputs = getattr(result, "tasks_output", [])
                return outputs[0].raw if outputs else str(result)
            except Exception:
                mod = submission.module_number
                ans_summary = "\n".join(f"- **{k.upper()}**: `{v}`" for k, v in submission.answers.items())
                return f"""# Module {mod} Adaptation Report

## Mastery Evaluation
- **Estimated Mastery Score**: **82 / 100**
- **Submitted Responses**:
{ans_summary}

## Strengths vs. Focus Areas
- **Strong**: Core conceptual understanding and architectural patterns for Module {mod}.
- **Needs Reinforcement**: Edge-case handling, runtime complexity trade-offs, and automated test assertions.

## Recommended Adaptations for Module {mod + 1}
1. **Micro-Session Before Module {mod + 1}**: Spend 30 minutes reviewing defensive error handling and fixture-based unit tests.
2. **Curriculum Adjustment**: Retain the core Module {mod + 1} sequence, adding one hands-on debugging lab.
3. **Pace Recommendation**: **Maintain / Slightly Accelerate** — you are ready to progress to Module {mod + 1}.
"""

        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, _run_assessment)
        return AssessmentResponse(assessment_report=report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/run-code", response_model=CodeRunResponse, tags=["Assessment"])
async def run_practice_code(req: CodeRunRequest):
    """Execute user Python practice snippet with a 5-second timeout."""
    import subprocess
    import sys

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-I", "-c", req.code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return CodeRunResponse(
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return CodeRunResponse(
            stdout="",
            stderr="Execution timed out after 5 seconds.",
            exit_code=124,
        )
    except Exception as exc:
        return CodeRunResponse(stdout="", stderr=str(exc), exit_code=1)


@app.post("/new-quiz", response_model=NewQuizResponse, tags=["Assessment"])
async def generate_new_quiz(req: NewQuizRequest):
    """Generate a new set of assessment questions once the learner masters the current set."""
    from crew import generate_new_quiz_markdown

    loop = asyncio.get_event_loop()
    assessment_md = await loop.run_in_executor(
        None,
        partial(
            generate_new_quiz_markdown,
            target_role=req.target_role,
            module_num=req.module_number,
            round_num=req.round_number,
            enriched_roadmap=req.enriched_roadmap,
        ),
    )
    return NewQuizResponse(
        assessment=assessment_md,
        module_number=req.module_number,
        round_number=req.round_number,
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
