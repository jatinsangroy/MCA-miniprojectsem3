"""
crew.py
-------
Assembles all agents and tasks into a cohesive CrewAI Crew
and exposes a run_pipeline() function that drives the full
end-to-end learning path generation workflow.
"""

from crewai import Crew, Process
from agents import (
    create_profiler_agent,
    create_curriculum_designer_agent,
    create_resource_curator_agent,
    create_assessment_adapter_agent,
)
from tasks import (
    create_profiling_task,
    create_curriculum_task,
    create_resource_task,
    create_assessment_task,
)


def run_pipeline(user_profile: dict, generate_quiz_for_module: int = 1) -> dict:
    """
    Execute the full 4-agent pipeline sequentially.

    Parameters
    ----------
    user_profile : dict
        {
            "name": str,
            "current_skills": list[str],
            "experience_years": int,
            "target_role": str,
            "weekly_hours": int
        }
    generate_quiz_for_module : int
        Which module number to generate an assessment for (default = 1).

    Returns
    -------
    dict with keys:
        "gap_report"      – output of Task 1
        "roadmap"         – output of Task 2
        "enriched_roadmap"– output of Task 3
        "assessment"      – output of Task 4
    """

    # --- Instantiate agents ---
    profiler = create_profiler_agent()
    curriculum_designer = create_curriculum_designer_agent()
    resource_curator = create_resource_curator_agent()
    assessment_adapter = create_assessment_adapter_agent()

    # --- Instantiate tasks (scoped context to minimize token usage) ---
    task_profile = create_profiling_task(user_profile, profiler)
    task_curriculum = create_curriculum_task(curriculum_designer)
    task_curriculum.context = [task_profile]

    task_resources = create_resource_task(resource_curator)
    task_resources.context = [task_curriculum]

    task_assessment = create_assessment_task(
        module_number=generate_quiz_for_module,
        assessment_agent=assessment_adapter,
    )
    task_assessment.context = [task_curriculum]

    # --- Assemble Crew ---
    crew = Crew(
        agents=[profiler, curriculum_designer, resource_curator, assessment_adapter],
        tasks=[task_profile, task_curriculum, task_resources, task_assessment],
        process=Process.sequential,  # Tasks run in order; output feeds context
        verbose=True,
    )

    # --- Kick off (with zero-token local fallback if API key is invalid/rate-limited) ---
    try:
        result = crew.kickoff()
    except Exception:
        return generate_fallback_pipeline(user_profile, generate_quiz_for_module)

    # CrewAI >=0.30 returns a CrewOutput object; extract individual task outputs
    task_outputs = result.tasks_output if hasattr(result, "tasks_output") else []

    def _get(index: int) -> str:
        try:
            return task_outputs[index].raw if task_outputs else str(result)
        except IndexError:
            return ""

    return {
        "gap_report": _get(0),
        "roadmap": _get(1),
        "enriched_roadmap": _get(2),
        "assessment": _get(3),
        "full_crew_output": str(result),
    }


def generate_fallback_pipeline(user_profile: dict, module_num: int = 1) -> dict:
    """Zero-token local generator when the LLM API key is invalid or rate-limited."""
    name = user_profile.get("name", "Learner")
    role = user_profile.get("target_role", "Software Engineer")
    skills = user_profile.get("current_skills") or ["Python"]
    skills_str = ", ".join(skills)
    exp = user_profile.get("experience_years", 0)
    hours = user_profile.get("weekly_hours", 10)
    weeks_per_mod = max(1, round(15 / max(hours, 1)))

    gap_report = f"""# Skill Gap Analysis for {name}

## Confirmed Strengths
- **Declared Foundation ({skills_str})**: Strong starting baseline for transitioning into a **{role}** track ({exp} yr(s) prior experience).
- **Study Commitment ({hours} hrs/week)**: Sufficient velocity to complete hands-on milestones every {weeks_per_mod}–{weeks_per_mod + 1} weeks.

## Critical Gaps (Ranked P1 → P4)
1. **P1 – Production Architecture & Core Patterns for {role}**: Moving from syntax/basics in {skills[0]} to modular, testable, production-grade application structure.
2. **P2 – Data Layer, APIs & Asynchronous Workflows**: Designing resilient REST/GraphQL services, query optimization, and schema migrations.
3. **P3 – Testing, Debugging & CI/CD Automation**: Automated unit/integration testing (`pytest`), containerization (`Docker`), and deployment pipelines.
4. **P4 – System Scalability, Security & Observability**: Caching, authentication/authorization (JWT/OAuth2), logging, and performance profiling.

## Nice-to-Have Gaps
- Cloud-native deployment orchestration (AWS/Render/Fly.io)
- Open-source contributions and technical design documentation

## Recommended Entry Point
Start immediately with **Module 1: Advanced {skills[0]} & Production Engineering Foundations** to bridge P1 before layering APIs and distributed systems.

## Summary Paragraph
{name} already brings a solid foothold in **{skills_str}**. By dedicating **{hours} hours/week**, {name} can systematically close all four critical hiring gaps for **{role}** across a structured 4-module curriculum culminating in a portfolio-ready capstone project.
"""

    roadmap = f"""# Learning Roadmap: {role}

## Module 1: Advanced {skills[0]} & Production Foundations
### Objective
Write clean, modular, type-safe, and idiomatic code tailored for production {role} systems.
### Topics
- Advanced data structures, generators, decorators, and context managers
- Type annotations, static analysis, and project packaging
- Concurrency fundamentals (`asyncio`, threading vs. multiprocessing)
### Estimated Duration
{weeks_per_mod}–{weeks_per_mod + 1} weeks ({hours} hrs/wk)
### Success Milestone
Build and package a CLI + modular library with 90%+ type coverage and clean architecture.

## Module 2: API Design, Data Persistence & State Management
### Objective
Design, build, and optimize data-driven backend services integrating {skills_str}.
### Topics
- RESTful API architecture, validation (Pydantic), and OpenAPI schemas
- Relational & NoSQL data modeling, indexing, and transactions
- Authentication, JWT session flows, and rate limiting
### Estimated Duration
{weeks_per_mod}–{weeks_per_mod + 1} weeks ({hours} hrs/wk)
### Success Milestone
Deploy a documented REST service with persistent database storage and auth middleware.

## Module 3: Automated Testing, Containers & CI/CD
### Objective
Guarantee reliability and repeatable deployments using modern DevOps workflows.
### Topics
- Unit, integration, and fixture-driven testing with `pytest`
- Multi-stage Dockerfiles and local orchestration with Docker Compose
- GitHub Actions CI pipelines for linting, testing, and automated builds
### Estimated Duration
{weeks_per_mod} weeks ({hours} hrs/wk)
### Success Milestone
Achieve 85%+ automated test coverage running automatically on every Git push.

## Module 4: Scalability, Observability & {role} Capstone
### Objective
Harden, monitor, and deploy a full-scale capstone application for {role} interviews.
### Topics
- Caching strategies (Redis), background task queues, and latency tuning
- Structured logging, health checks, and error tracking
- End-to-end system design and live cloud deployment
### Estimated Duration
2 weeks ({hours} hrs/wk)
### Success Milestone
Ship the live Capstone Project with public documentation, benchmarks, and architecture diagram.

## Capstone Project
Build and deploy a production-ready **{role} Platform** combining **{skills_str}**, a documented API backend, automated test suite, Docker containerization, and live cloud hosting.
"""

    enriched_roadmap = f"""# Enriched Learning Roadmap: {role}

## Module 1: Advanced {skills[0]} & Production Foundations
### Objective
Write clean, modular, type-safe, and idiomatic code tailored for production {role} systems.
### Resources
1. **Best Free Resource**: *Real Python – Core & Advanced Tutorials* (https://realpython.com) — Real Python, ~12 hrs, Intermediate.
2. **Best Paid Resource**: *Fluent Python (2nd Edition) by Luciano Ramalho* (https://www.oreilly.com) — O'Reilly ($45), ~25 hrs, Intermediate/Advanced.
3. **Hands-On Exercise**: Build an asynchronous task runner with custom retry decorators and full `mypy` type checking (~4 hrs).
4. **Official Docs Reference**: *Official Python Documentation – Data Model & typing* (https://docs.python.org/3/reference/datamodel.html).

## Module 2: API Design, Data Persistence & State Management
### Objective
Design, build, and optimize data-driven backend services integrating {skills_str}.
### Resources
1. **Best Free Resource**: *FastAPI Official Tutorial – User Guide* (https://fastapi.tiangolo.com/tutorial/) — FastAPI Docs, ~10 hrs, Beginner/Intermediate.
2. **Best Paid Resource**: *Test-Driven Development with Python & FastAPI* (https://testdriven.io) — TestDriven.io ($40), ~15 hrs, Intermediate.
3. **Hands-On Exercise**: Create a CRUD + JWT-authenticated REST API backed by SQLite/PostgreSQL with indexed queries (~6 hrs).
4. **Official Docs Reference**: *SQLAlchemy 2.0 Unified Tutorial* (https://docs.sqlalchemy.org/en/20/tutorial/).

## Module 3: Automated Testing, Containers & CI/CD
### Objective
Guarantee reliability and repeatable deployments using modern DevOps workflows.
### Resources
1. **Best Free Resource**: *Full Stack Python – Testing & Deployment* (https://www.fullstackpython.com/testing.html) — Web Guide, ~8 hrs, Intermediate.
2. **Best Paid Resource**: *Python Testing with pytest (2nd Edition) by Brian Okken* (https://pragprog.com/titles/bopytest2/) — Pragmatic Bookshelf ($35), ~14 hrs, Intermediate.
3. **Hands-On Exercise**: Containerize your Module 2 API with a slim Dockerfile and add a GitHub Actions workflow running `pytest` (~4 hrs).
4. **Official Docs Reference**: *pytest Documentation – Fixtures & Parametrization* (https://docs.pytest.org/en/stable/).

## Module 4: Scalability, Observability & {role} Capstone
### Objective
Harden, monitor, and deploy a full-scale capstone application for {role} interviews.
### Resources
1. **Best Free Resource**: *System Design Primer* (https://github.com/donnemartin/system-design-primer) — GitHub, ~15 hrs, Intermediate/Advanced.
2. **Best Paid Resource**: *Designing Data-Intensive Applications by Martin Kleppmann* (https://dataintensive.net) — O'Reilly ($50), ~30 hrs, Advanced.
3. **Hands-On Exercise**: Add caching, rate limiting, `/health` telemetry, and deploy your Capstone service to Render or Fly.io (~8 hrs).
4. **Official Docs Reference**: *12-Factor App Methodology* (https://12factor.net).
"""

    assessment = f"""# Module {module_num} Assessment

## Quiz
### Q1 (MCQ)
Which mechanism best prevents resource leaks (such as open files or database connections) even when an exception occurs?
- A) Global variables
- B) Context managers (`with` statement / `__enter__` & `__exit__`)
- C) Recursive functions
- D) Implicit garbage collection only

### Q2 (MCQ)
When designing a production service for a **{role}**, why are lazy generators (`yield`) preferred over returning large lists in memory?
- A) They automatically parallelize CPU cores
- B) They yield items on demand with $O(1)$ auxiliary memory overhead
- C) They bypass syntax checking
- D) They encrypt data in transit

### Q3 (MCQ)
What is the primary benefit of combining static type hints with runtime schema validation (e.g., Pydantic)?
- A) It eliminates the need for a database
- B) It catches type mismatches early in IDE/CI and guarantees valid payloads at runtime boundaries
- C) It makes network cables faster
- D) It replaces version control

### Q4 (Short Answer)
Explain the difference between I/O-bound concurrency (`asyncio`) and CPU-bound parallelism (`multiprocessing`) in 2–3 sentences.

### Q5 (Short Answer)
How does dependency injection improve unit testing when building a **{role}** backend service?

### Q6 (Code)
Complete the context-managed timer decorator snippet below:
```python
import time
from functools import wraps

def log_execution_time(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = ________(*args, **kwargs)  # Call wrapped function
        elapsed = time.perf_counter() - start
        print(f"{{fn.__name__}} took {{elapsed:.3f}}s")
        return result
    return wrapper
```

### Q7 (Code)
Fix the mutable default argument bug in the function below:
```python
def append_skill(skill: str, skills: list[str] = []):
    skills.append(skill)
    return skills
```

## Answer Key
- **Q1**: **B** — Context managers guarantee cleanup via `__exit__` even if exceptions are raised.
- **Q2**: **B** — Generators stream elements lazily without allocating the entire collection in RAM.
- **Q3**: **B** — Static typing + runtime validation protects both developer workflows and external API boundaries.
- **Q4**: `asyncio` uses a single-threaded event loop to overlap waiting on network/disk I/O, whereas `multiprocessing` spawns separate OS processes to run CPU-heavy computations across multiple cores.
- **Q5**: It decouples business logic from external clients (databases, APIs), allowing tests to pass lightweight mocks or in-memory fakes without network calls.
- **Q6**: Replace blank with `fn`.
- **Q7**: Default to `skills: list[str] | None = None` and initialize `if skills is None: skills = []` inside the function.
"""

    return {
        "gap_report": gap_report,
        "roadmap": roadmap,
        "enriched_roadmap": enriched_roadmap,
        "assessment": assessment,
        "full_crew_output": assessment,
    }


def generate_new_quiz_markdown(
    target_role: str = "Python Developer",
    module_num: int = 1,
    round_num: int = 2,
    enriched_roadmap: str = "",
) -> str:
    """
    Generate a fresh 7-question quiz using GROQ_ASSESSMENT_API_KEY,
    falling back to rotating question banks if the API is unavailable/rate-limited.
    """
    try:
        agent = create_assessment_adapter_agent()
        task = create_assessment_task(
            module_number=module_num,
            assessment_agent=agent,
        )
        task.description = (
            f"Context Roadmap:\n{enriched_roadmap[:1500]}\n\n"
            f"Generate Round {round_num} (a brand-new set of 7 unseen questions) for Module {module_num} "
            f"tailored to a {target_role}.\n" + task.description
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
        )
        result = crew.kickoff()
        outputs = getattr(result, "tasks_output", [])
        raw = outputs[0].raw if outputs else str(result)
        if "## Quiz" in raw and "### Q1" in raw:
            return raw
    except Exception:
        pass

    # Rotating fallback question sets so each round has fresh questions
    bank_index = (round_num - 1) % 3
    if bank_index == 1:
        return f"""# Module {module_num} Assessment (Round {round_num})

## Quiz
### Q1 (MCQ)
Which HTTP status code should a REST API return when a new resource is successfully created via `POST`?
- A) 200 OK
- B) 201 Created
- C) 204 No Content
- D) 301 Moved Permanently

### Q2 (MCQ)
In database indexing for a **{target_role}** application, what is the main trade-off of adding too many secondary B-tree indexes to a table?
- A) `SELECT` queries become slower
- B) `INSERT`, `UPDATE`, and `DELETE` operations incur extra write overhead and storage cost
- C) Primary keys stop working
- D) Transactions can no longer roll back

### Q3 (MCQ)
Which `pytest` feature allows running the same test function across multiple input/output pairs without duplicating test code?
- A) `@pytest.mark.parametrize`
- B) `@pytest.fixture(autouse=True)`
- C) `unittest.SkipTest`
- D) `sys.exit()`

### Q4 (Short Answer)
Why should secrets (like `GROQ_API_KEY` or database passwords) be loaded from environment variables rather than hardcoded in source files?

### Q5 (Short Answer)
What problem does an API rate limiter (such as token bucket) solve in production services?

### Q6 (Code)
Complete the dictionary comprehension below to map each skill name to its character length:
```python
skills = ["python", "sql", "fastapi"]
skill_lengths = {{s: ______(s) for s in skills}}
print(skill_lengths)
```

### Q7 (Code)
Fix the SQL injection vulnerability in the query execution below using parameterized placeholders:
```python
def get_user_by_email(cursor, email: str):
    # Fix unsafe string interpolation:
    cursor.execute("SELECT id, name FROM users WHERE email = ?", (email,))
    return cursor.fetchone()
```

## Answer Key
- **Q1**: **B** — `201 Created` specifically indicates that a request succeeded and resulted in a new resource being created.
- **Q2**: **B** — Every write operation must also update all affected B-tree indexes, increasing write latency and disk usage.
- **Q3**: **A** — `@pytest.mark.parametrize` executes the same test body with multiple argument sets cleanly.
- **Q4**: Storing secrets in environment variables prevents accidental credential leaks in Git history and allows per-environment rotation without code changes.
- **Q5**: Rate limiting protects backend services from abuse, DDoS spikes, and cascading resource exhaustion by capping requests per client window.
- **Q6**: Use `len(s)` (i.e. `{s: len(s) for s in skills}`).
- **Q7**: Pass parameters as a tuple `("SELECT id, name FROM users WHERE email = ?", (email,))` so the DB driver safely escapes input.
"""
    elif bank_index == 2:
        return f"""# Module {module_num} Assessment (Round {round_num})

## Quiz
### Q1 (MCQ)
What is the time complexity of looking up a key in a Python `dict` (hash table) on average?
- A) O(n)
- B) O(log n)
- C) O(1)
- D) O(n log n)

### Q2 (MCQ)
In Docker, why should `COPY requirements.txt .` and `RUN pip install -r requirements.txt` appear *before* `COPY . .`?
- A) Docker requires alphabetical ordering of commands
- B) It leverages Docker layer caching so dependencies aren't reinstalled on every source code edit
- C) Python refuses to run otherwise
- D) It disables network access

### Q3 (MCQ)
Which principle states that a class or module should have only one reason to change?
- A) Single Responsibility Principle (SRP)
- B) Liskov Substitution Principle
- C) Premature Optimization Principle
- D) Global State Principle

### Q4 (Short Answer)
Explain what an idempotent HTTP method is and give one example (`GET`, `PUT`, `DELETE`, vs `POST`).

### Q5 (Short Answer)
When debugging a slow endpoint, how do you distinguish between an N+1 database query problem and a CPU-bound serialization bottleneck?

### Q6 (Code)
Complete the `async` function below so both tasks run concurrently using `asyncio.gather`:
```python
import asyncio

async def fetch_profile(uid: int) -> dict:
    await asyncio.sleep(0.01)
    return {{"uid": uid}}

async def load_two(u1: int, u2: int):
    return await asyncio.______(fetch_profile(u1), fetch_profile(u2))
```

### Q7 (Code)
Write a function `filter_even(nums: list[int]) -> list[int]` that returns only the even integers from `nums`:
```python
def filter_even(nums: list[int]) -> list[int]:
    return [n for n in nums if n % 2 == 0]
```

## Answer Key
- **Q1**: **C** — Hash table lookups run in amortized $O(1)$ average time.
- **Q2**: **B** — Separating dependency installation into an earlier cached layer avoids rebuilding packages when application files change.
- **Q3**: **A** — The Single Responsibility Principle (SRP) keeps modules focused, cohesive, and easier to maintain.
- **Q4**: An idempotent operation produces the same server state whether executed once or multiple times (e.g., `GET`, `PUT`, `DELETE`, whereas `POST` typically creates duplicates).
- **Q5**: Query logging/tracing reveals dozens of repeated SQL statements for N+1 issues, whereas CPU profiling shows time spent inside JSON/model serialization on a single query result.
- **Q6**: Use `asyncio.gather(fetch_profile(u1), fetch_profile(u2))`.
- **Q7**: Use `[n for n in nums if n % 2 == 0]`.
"""
    else:
        return generate_fallback_pipeline(
            {"target_role": target_role}, module_num=module_num
        )["assessment"]
