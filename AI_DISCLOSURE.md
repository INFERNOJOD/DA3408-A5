# AI Usage Disclosure — DA3408 Module 4 (Krish Yadav, DA24B043)

## AI Tools Used

- Claude (Anthropic)
- ChatGPT (OpenAI)

## How They Were Used and Impact

1. **Conceptual Clarification and Design Guidance**
   - Used AI to discuss implementation approaches and clarify concepts across the four parts, including FastAPI validation and middleware, Prometheus metrics, load testing, Little's Law, MLflow model registration and aliases, promotion gates, shadow testing, Ray Serve deployments, sticky canaries and release smoke tests.
   - Discussed design choices, implementation trade-offs and how the different stages of the deployment pipeline fit together.
   - **Impact:** Helped clarify architectural decisions and implementation details before and during development.

2. **Code Generation and Implementation Assistance**
   - Used AI assistance for selected portions of the implementation across `partA/` to `partD/`, including API handling, validation, testing utilities, MLflow workflows, deployment configuration and release checks.
   - AI also assisted with selected code modifications, including resolving the `ETA_MODEL` path against the repository root in `eta_app.py` and creating the read-only `partB/list_registry.py` utility.
   - Reviewed the generated suggestions and adapted them to the existing implementation and project requirements.
   - **Impact:** Reduced implementation and debugging time and helped speed up development across the different components.

3. **Debugging, Testing and Performance Analysis**
   - Used AI to review code, identify potential edge cases, troubleshoot implementation issues and inspect test results.
   - Discussed load-test behaviour across different concurrency levels, batch sizes and worker configurations, including saturation, throughput, latency and capacity.
   - Used AI assistance to review MLflow promotion logic, shadow comparisons, Ray Serve deployment behaviour and smoke-test results.
   - **Impact:** Helped identify potential issues, improve test coverage and interpret experimental results.

4. **Report, Documentation and Demonstration Assistance**
   - Used AI to review the README against the project requirements and suggest potential improvements. The README itself was written entirely by me.
   - Wrote the technical answers and explanations in `REPORT.md` myself, building on the provided skeleton and using my own experimental outputs and results. AI was used to review selected sections, suggest improvements to clarity and check technical wording.
   - Used AI assistance for organising tables and refining the presentation of the report.
   - **Impact:** Helped identify documentation issues, improve clarity and organise the presentation without replacing my own technical explanations and analysis.

## Contribution and Responsibility

- I was responsible for integrating the components, running the services and experiments, evaluating the results and making the final engineering decisions.
- All load tests, model-serving experiments, Ray Serve checks and release verification workflows were executed on my own Ubuntu ARM64 VM (4 vCPUs, 7.2 GiB RAM).
- All 16 screenshots and experimental output files were collected from actual runs. The numbers reported in `REPORT.md` are based on these outputs.
- I reviewed the final implementation and documentation and take responsibility for the submitted code and results.

Lecture-notebook reuse is acknowledged separately under Citations in `REPORT.md`.
