# Robot Agent `src` 개요

![LangGraph 전체 흐름](../graph.png)

## 코드 구성
- `common/`·`config/`: 모델 이름, 예외, 로깅 유틸과 Pydantic 기반 설정(`config.yaml`, `Config`, `NodeConfig`, `RunnerConfig`)을 로드하는 코드.
- `prompts/`: 계획 프롬프트(`planning_prompt.py`: 목표/태스크 분해)와 프로세스 프롬프트(`process_prompt.py`: 의도 분기 → 감독 → 피드백)의 템플릿·파서·입력 가공 함수.
- `runner/`: LangGraph 노드/그래프 생성기(`graph.py`), 기본 `Runner`와 `PlanRunner`/`SupervisedPlanRunner`, 상태 스키마 및 환경 정보를 텍스트로 만드는 `state.py`·`text.py`.
- `utils/`: 파일 입출력 헬퍼(`load`, `save`).  
  `rag/`, `tools/`는 RAG·툴 연동용 자리표시자.

## 주요 플로우
1. `StateMaker.make(user_query)`에서 사용자 질의와 환경 정보(오브젝트/그룹 목록, 스킬 텍스트)를 담은 초기 상태를 생성합니다(`runner/text.py`가 REST 엔드포인트 `http://127.0.0.1:8800/env_entire`를 조회).
2. `SupervisedPlanRunner.build_graph()`는 LangGraph를 구성합니다.  
   - `USER_INPUT → INTENT`(stop/accept/new 분기) → `SUPERVISOR`(최종 질의·feasibility 판정) → `FEEDBACK`(불가 시 수정 제안) → `GOAL_DECOMP`(상위 목표 분해) → `TASK_DECOMP`(스킬 단위 태스크 나열).
3. `runner.invoke(state)`를 호출하면 그래프가 실행되어 상태에 `intent_result`, `supervisor_result`, `feedback_result`, `subgoals`, `tasks`가 채워집니다.
4. `graph.get_graph().draw_mermaid_png()`를 통해 위 흐름을 `graph.png`로 저장할 수 있습니다(이미지 경로는 README 상단 참조).

## `test_planning.ipynb` 동작 요약
1. `load_config()`로 설정을 불러오고, `StateMaker`로 예시 질의(예: `"put a fork on the island table"`) 상태를 만듭니다.
2. `SupervisedPlanRunner`를 생성해 `runner.invoke(state)`로 LangGraph 파이프라인을 한 번 실행합니다.
3. 실행 결과를 포함한 그래프 객체에서 `draw_mermaid_png()`를 호출해 계획 그래프를 `graph.png`로 저장합니다. 원하는 질의로 셀을 수정하면 동일한 플로우로 재실행할 수 있습니다.
