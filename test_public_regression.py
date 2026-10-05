"""Public, deterministic regression tests using synthetic resume data only."""

import unittest
from unittest.mock import Mock, patch

import pymupdf
from fastapi.testclient import TestClient

import api
from agent_loop import (
    AgentState,
    analyze_history,
    authoritative_employer_result,
    get_candidate_skills,
    answer_contradicts_evidence,
    answer_has_unsupported_positive_claims,
    answer_has_unsupported_relationship_claims,
    complete_answer_with_relevant_evidence,
    ground_final_answer,
    MODEL_NAME,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_URL,
    OLLAMA_WARMUP_PROMPT,
    OLLAMA_WARMUP_TIMEOUT,
    call_ollama,
    run_agent,
    search_resume,
    warm_up_ollama,
    _normalize_tokens,
    QUERY_META_WORDS,
    SEARCH_STOPWORDS,
)
from resume_parser import parse_resume


SYNTHETIC_RESUME = """Alex Rivera
alex@example.com
EXPERIENCE
Example Software Inc.
Software Engineer
Built internal APIs using Python and PostgreSQL.
Maintained the deployment dashboard for the engineering team.
SKILLS
Python, React, PostgreSQL, Docker, Flutter
"""

EXPERIENCE_EVIDENCE = (
    "Example Software Inc.\n"
    "Software Engineer\n"
    "Built internal APIs using Python and PostgreSQL.\n"
    "Maintained the deployment dashboard for the engineering team."
)
SKILLS_EVIDENCE = "Python, React, PostgreSQL, Docker, Flutter"
NO_NAME_RESUME = """EDUCATION
Bachelor of Science in Information Systems
Example University
"""

EDUCATION_INTENT_RESUME = """Alex Rivera
EDUCATION
Bachelor of Science in Information Systems
Example University
"""

PROJECT_SCOPING_RESUME = """Alex Rivera
PROJECT EXPERIENCE
CAPSTONE PROJECT MANAGER
Example Food Delivery App
Built the app using ProjectOnlyFramework.

OTHER PROJECT
Technologies in this separate project: OtherOnlyFramework.

EXPERIENCE
Example Corporation
Software Intern
Built an internal system unrelated to the capstone.

SKILLS
GlobalOnlyFramework
Python
"""

GENERIC_SKILL_LABEL_RESUME = """Casey Morgan
SKILLS
Programming: LanguageAlpha, LanguageBeta
Frameworks: FrameworkAlpha (PluginAlpha, PluginBeta)
Databases: StoreAlpha
Tools: ToolAlpha
Languages: LanguageGamma
Cloud Platforms: PlatformAlpha
"""

PROJECT_RELATIONSHIP_RESUME = """Casey Morgan
PROJECT EXPERIENCE
CAPSTONE PROJECT
Example Delivery Platform
Built the platform using CapstoneOnlyFramework.

OTHER PROJECT
Example Inventory Tool
Built the inventory tool using OtherProjectOnlyFramework.

EXPERIENCE
Example Corporation
Software Intern
Used ExperienceOnlyFramework in internal applications.

SKILLS
Frameworks: GlobalOnlyFramework
"""


def state_with_tool_results(question, results, resume_text=SYNTHETIC_RESUME):
    messages = [{"role": "user", "content": question}]
    for category, evidence in results:
        messages.append({
            "role": "tool",
            "tool": "search_resume",
            "arguments": {"query": question},
            "result": {"category": category, "results": [evidence]},
        })
    return AgentState(messages=messages, resume_text=resume_text)


def make_synthetic_pdf():
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(72, 72, page.rect.width - 72, page.rect.height - 72),
        SYNTHETIC_RESUME,
        fontsize=11,
    )
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class GroundingRegressionTests(unittest.TestCase):
    def test_supported_relationship_within_same_experience_entry_is_accepted(self):
        question = "Did the candidate use Python at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [("experience", EXPERIENCE_EVIDENCE)],
        )
        analysis = analyze_history(state.messages, state.resume_text)
        groups = analysis["evidence_groups"]
        experience_group = next(group for group in groups if group["section"] == "experience")
        claim = "The candidate used Python at Example Software Inc."

        self.assertEqual(experience_group["entry_id"], "experience:0")
        self.assertFalse(answer_has_unsupported_relationship_claims(claim, analysis))
        self.assertFalse(answer_contradicts_evidence(claim, analysis))

    def test_skill_listed_separately_does_not_support_employer_relationship(self):
        question = "Did the candidate use React at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [
                ("experience", EXPERIENCE_EVIDENCE),
                ("skills", SKILLS_EVIDENCE),
            ],
        )
        analysis = analyze_history(state.messages, state.resume_text)
        group_ids = {group["entry_id"] for group in analysis["evidence_groups"]}
        claim = "The candidate used React at Example Software Inc."

        self.assertIn("experience:0", group_ids)
        self.assertIn("skills:0", group_ids)
        self.assertTrue(answer_has_unsupported_relationship_claims(claim, analysis))
        self.assertTrue(answer_contradicts_evidence(claim, analysis))

    def test_unsupported_technology_claim_is_rejected(self):
        question = "Does Alex know Kotlin?"
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": "Kotlin"},
                "result": {
                    "category": "skills",
                    "results": [],
                    "not_found_terms": ["Kotlin"],
                },
            },
        ], resume_text=SYNTHETIC_RESUME)
        analysis = analyze_history(state.messages, state.resume_text)
        claim = "Alex used Kotlin to build mobile apps."

        self.assertTrue(answer_has_unsupported_positive_claims(claim, analysis))
        self.assertTrue(answer_contradicts_evidence(claim, analysis))

    def test_favorite_language_remains_unsupported(self):
        question = "What is the candidate's favorite programming language?"
        result = search_resume(question, SYNTHETIC_RESUME)
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ], resume_text=SYNTHETIC_RESUME)
        analysis = analyze_history(state.messages, state.resume_text)
        claim = "Python is the candidate's favorite programming language."

        self.assertIn("favorite", result.get("not_found_terms", []))
        self.assertTrue(answer_has_unsupported_positive_claims(claim, analysis))
        self.assertTrue(answer_contradicts_evidence(claim, analysis))

    def test_flutter_skill_does_not_support_company_relationship(self):
        question = "Did the candidate use Flutter at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [
                ("experience", EXPERIENCE_EVIDENCE),
                ("skills", SKILLS_EVIDENCE),
            ],
        )
        analysis = analyze_history(state.messages, state.resume_text)
        claim = "The candidate used Flutter at Example Software Inc."

        self.assertTrue(answer_has_unsupported_relationship_claims(claim, analysis))
        self.assertTrue(answer_contradicts_evidence(claim, analysis))

    def test_completeness_adds_separately_supported_responsibilities(self):
        question = "What responsibilities did Alex have?"
        state = state_with_tool_results(
            question,
            [("experience", EXPERIENCE_EVIDENCE)],
        )
        partial_answer = "The candidate built internal APIs."

        completed = complete_answer_with_relevant_evidence(partial_answer, state)

        self.assertIn("built internal apis", completed.lower())
        self.assertIn("maintained the deployment dashboard", completed.lower())

    def test_unsafe_relationship_falls_back_to_resume_evidence(self):
        question = "Did Alex use React at Example Software Inc.?"
        state = state_with_tool_results(
            question,
            [
                ("experience", EXPERIENCE_EVIDENCE),
                ("skills", SKILLS_EVIDENCE),
            ],
        )
        unsupported_answer = "The candidate used React at Example Software Inc."

        with patch("agent_loop.ask_llm_for_decision", return_value={
            "action": "final",
            "answer": unsupported_answer,
        }):
            fallback = ground_final_answer(state, unsupported_answer)

        self.assertNotIn(unsupported_answer, fallback)
        self.assertIn("Built internal APIs using Python and PostgreSQL", fallback)
        self.assertIn("React", fallback)
        self.assertIn("does not establish", fallback.lower())


class EducationGroundingTests(unittest.TestCase):
    def test_university_attendance_word_is_contextually_grounded(self):
        question = "What university did the candidate attend?"
        result = search_resume(question, EDUCATION_INTENT_RESUME)
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ], resume_text=EDUCATION_INTENT_RESUME)

        self.assertNotIn("attend", result.get("not_found_terms", []))
        answer = ground_final_answer(state, "The candidate attended Example University.")

        self.assertEqual(answer, "The candidate attended Example University.")
        self.assertTrue(state.grounding_verified)

    def test_education_context_does_not_allow_unsupported_claims(self):
        question = "What university did the candidate attend?"
        result = search_resume(question, EDUCATION_INTENT_RESUME)
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ], resume_text=EDUCATION_INTENT_RESUME)
        analysis = analyze_history(state.messages, state.resume_text)

        unsupported = "The candidate attended Example University and worked at Fake Corporation."
        self.assertTrue(answer_has_unsupported_positive_claims(unsupported, analysis))
        self.assertTrue(answer_contradicts_evidence(unsupported, analysis))

        missing_result = {
            "category": "education",
            "results": [],
            "not_found_terms": ["university"],
        }
        missing_state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": missing_result,
            },
        ], resume_text="Alex Rivera\nSKILLS\nPython\n")
        missing_analysis = analyze_history(missing_state.messages, missing_state.resume_text)
        self.assertTrue(
            answer_has_unsupported_positive_claims("The candidate attended.", missing_analysis)
        )


class EmployerResponsibilityIntentTests(unittest.TestCase):
    def test_absent_employer_uses_authoritative_preflight_before_tool_selection(self):
        question = "What were the candidate's responsibilities at Example Corporation?"
        resume = """Alex Rivera
EXPERIENCE
Different Employer
Operations Associate
Handled scheduling.
PROJECTS
Example Project
Built an example.
"""
        authoritative_result = authoritative_employer_result(question, resume)

        self.assertIsNotNone(authoritative_result)
        self.assertEqual(authoritative_result["employer_check"], "not_found")
        self.assertEqual(authoritative_result["results"], [])

        def final_from_authoritative_verdict(state, **_kwargs):
            search_calls = [
                message for message in state.messages
                if message["role"] == "tool" and message.get("tool") == "search_resume"
            ]
            self.assertEqual(len(search_calls), 1)
            self.assertEqual(search_calls[0]["result"]["employer_check"], "not_found")
            self.assertEqual(state.employer_verdict["employer_check"], "not_found")
            return {
                "action": "final",
                "answer": "The resume provides no evidence.",
            }

        with patch("agent_loop.ask_llm_for_decision", side_effect=final_from_authoritative_verdict) as ask_llm:
            outcome = run_agent(question, resume)

        self.assertEqual(outcome["tools_executed"], 1)
        self.assertEqual(outcome["iterations"], 1)
        self.assertEqual(outcome["tools_refused"], 0)
        ask_llm.assert_called_once()
        self.assertIn("no evidence", outcome["answer"].lower())

    def test_present_employer_responsibilities_returns_matching_entry(self):
        question = "What were the candidate's responsibilities at Example Corporation?"
        result = search_resume(question, PROJECT_SCOPING_RESUME)

        self.assertIn("Software Intern", "\n".join(result["results"]))
        self.assertIn("Built an internal system unrelated to the capstone", "\n".join(result["results"]))
        self.assertNotIn("employer_check", result)

    def test_project_location_is_not_employer_intent(self):
        question = "What were the candidate's responsibilities in Example Project?"
        resume = """Alex Rivera
EXPERIENCE
Example Corporation
Operations Associate
Handled scheduling.
PROJECTS
Example Project
Built an example.
"""

        self.assertIsNone(authoritative_employer_result(question, resume))

    def test_duties_at_entity_uses_authoritative_employer_lookup(self):
        question = "What were the candidate's duties at Missing Corporation?"
        resume = "Alex Rivera\nEXPERIENCE\nExample Corporation\nOperations Associate\nHandled scheduling.\n"
        result = authoritative_employer_result(question, resume)

        self.assertIsNotNone(result)
        self.assertEqual(result["employer_check"], "not_found")


class ProjectTechnologyScopingTests(unittest.TestCase):
    def test_project_technology_query_uses_matching_project_not_global_skills(self):
        question = "What technologies did the candidate use in their capstone project?"
        result = search_resume(question, PROJECT_SCOPING_RESUME)
        evidence = "\n".join(result["results"])
        query_terms = set(_normalize_tokens(question)) - SEARCH_STOPWORDS - QUERY_META_WORDS
        project_entries = parse_resume(PROJECT_SCOPING_RESUME)["projects"]
        flat_scores = [
            len(query_terms & set(_normalize_tokens(entry["raw_text"])))
            for entry in project_entries
        ]

        self.assertEqual(flat_scores, [2, 2])
        self.assertIn("ProjectOnlyFramework", evidence)
        self.assertNotIn("GlobalOnlyFramework", evidence)
        self.assertNotIn("internal system unrelated to the capstone", evidence)
        self.assertNotIn("OTHER PROJECT", evidence)

    def test_general_technology_question_still_returns_skills(self):
        result = search_resume(
            "What technologies does the candidate know?",
            PROJECT_SCOPING_RESUME,
        )

        self.assertIn("GlobalOnlyFramework", "\n".join(result["results"]))


class GenericSkillsLabelTests(unittest.TestCase):
    def test_arbitrary_skill_labels_are_parsed_without_technology_vocabulary(self):
        skills = get_candidate_skills(GENERIC_SKILL_LABEL_RESUME)

        self.assertEqual(
            skills["programming_languages"],
            ["LanguageAlpha", "LanguageBeta", "LanguageGamma"],
        )
        self.assertIn("FrameworkAlpha (PluginAlpha, PluginBeta)", skills["frameworks_and_technologies"])
        self.assertIn("StoreAlpha", skills["frameworks_and_technologies"])
        self.assertIn("PlatformAlpha", skills["frameworks_and_technologies"])
        self.assertEqual(skills["development_tools"], ["ToolAlpha"])

    def test_labeled_database_is_retrieved_and_general_skills_remain_available(self):
        database_result = search_resume(
            "What database is listed in the candidate's skills?",
            GENERIC_SKILL_LABEL_RESUME,
        )
        general_result = search_resume(
            "What technologies does the candidate know?",
            GENERIC_SKILL_LABEL_RESUME,
        )

        self.assertIn("StoreAlpha", "\n".join(database_result["results"]))
        self.assertIn("PlatformAlpha", "\n".join(general_result["results"]))


class ProjectEntryGroundingTests(unittest.TestCase):
    def _analysis_for(self, question):
        result = search_resume(question, PROJECT_RELATIONSHIP_RESUME)
        messages = [
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ]
        return analyze_history(messages, PROJECT_RELATIONSHIP_RESUME)

    def test_named_project_retrieval_prefers_project_over_global_skills(self):
        result = search_resume(
            "What framework was used in the Example Inventory Tool?",
            PROJECT_RELATIONSHIP_RESUME,
        )
        evidence = "\n".join(result["results"])

        self.assertIn("OtherProjectOnlyFramework", result["results"][0])
        self.assertNotIn("GlobalOnlyFramework", evidence)
        self.assertNotIn("CapstoneOnlyFramework", evidence)

    def test_unmatched_named_project_does_not_fall_back_to_another_project(self):
        result = search_resume(
            "What framework was used in the Example Missing Tool?",
            PROJECT_RELATIONSHIP_RESUME,
        )
        evidence = "\n".join(result["results"])

        self.assertNotIn("CapstoneOnlyFramework", evidence)
        self.assertNotIn("OtherProjectOnlyFramework", evidence)
        self.assertNotIn("GlobalOnlyFramework", evidence)

    def test_technology_in_target_project_supports_relationship(self):
        question = "Did the candidate use CapstoneOnlyFramework in the capstone project?"
        answer = "The candidate used CapstoneOnlyFramework in the capstone project."
        analysis = self._analysis_for(question)

        self.assertFalse(answer_has_unsupported_relationship_claims(answer, analysis))

    def test_skills_only_technology_does_not_support_project_relationship(self):
        question = "Did the candidate use GlobalOnlyFramework in the capstone project?"
        answer = "The candidate used GlobalOnlyFramework in the capstone project."
        analysis = self._analysis_for(question)

        self.assertTrue(answer_has_unsupported_relationship_claims(answer, analysis))

    def test_other_project_technology_does_not_support_capstone_relationship(self):
        question = "Did the candidate use OtherProjectOnlyFramework in the capstone project?"
        answer = "The candidate used OtherProjectOnlyFramework in the capstone project."
        analysis = self._analysis_for(question)

        self.assertTrue(answer_has_unsupported_relationship_claims(answer, analysis))

    def test_experience_technology_does_not_support_project_relationship(self):
        question = "Did the candidate use ExperienceOnlyFramework in the capstone project?"
        answer = "The candidate used ExperienceOnlyFramework in the capstone project."
        analysis = self._analysis_for(question)

        self.assertTrue(answer_has_unsupported_relationship_claims(answer, analysis))

    def test_prior_denial_does_not_hide_later_unsupported_project_claim(self):
        question = "Did the candidate use GlobalOnlyFramework in the capstone project?"
        answer = (
            "The resume does not establish that relationship, but the candidate used "
            "GlobalOnlyFramework in the capstone project."
        )
        analysis = self._analysis_for(question)

        self.assertTrue(answer_has_unsupported_relationship_claims(answer, analysis))

    def test_grounding_replaces_unsupported_project_relationship_with_safe_fallback(self):
        question = "Did the candidate use OtherProjectOnlyFramework in the capstone project?"
        result = search_resume(question, PROJECT_RELATIONSHIP_RESUME)
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ], resume_text=PROJECT_RELATIONSHIP_RESUME)
        unsupported_answer = (
            "The candidate used OtherProjectOnlyFramework in the capstone project."
        )

        with patch("agent_loop.ask_llm_for_decision", return_value={
            "action": "final",
            "answer": unsupported_answer,
        }):
            safe_answer = ground_final_answer(state, unsupported_answer)

        self.assertNotIn(unsupported_answer, safe_answer)
        self.assertIn("does not establish", safe_answer.lower())
        self.assertFalse(state.grounding_verified)

    def test_fallback_names_the_unestablished_project_technology_relationship(self):
        question = "Was OtherProjectOnlyFramework used in the capstone project?"
        result = search_resume(question, PROJECT_RELATIONSHIP_RESUME)
        state = AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ], resume_text=PROJECT_RELATIONSHIP_RESUME)
        answer = (
            "According to the resume: CAPSTONE PROJECT uses CapstoneOnlyFramework; "
            "OTHER PROJECT uses OtherProjectOnlyFramework."
        )

        with patch("agent_loop.ask_llm_for_decision", return_value={
            "action": "final",
            "answer": answer,
        }):
            grounded_answer = ground_final_answer(state, answer)

        self.assertIn("does not establish", grounded_answer.lower())
        self.assertIn("OtherProjectOnlyFramework", grounded_answer)
        self.assertIn("capstone project", grounded_answer.lower())
        self.assertFalse(state.grounding_verified)


class FavoriteLanguageGroundingTests(unittest.TestCase):
    def _favorite_state(self, resume):
        question = "What is the candidate's favorite programming language?"
        result = search_resume(question, resume)
        return AgentState(messages=[
            {"role": "user", "content": question},
            {
                "role": "tool",
                "tool": "search_resume",
                "arguments": {"query": question},
                "result": result,
            },
        ], resume_text=resume)

    def test_listed_languages_without_favorite_get_explicit_refusal(self):
        resume = """Casey Morgan
SKILLS
Programming: LanguageAlpha, LanguageBeta
"""
        state = self._favorite_state(resume)
        draft = "According to the resume: Programming languages: LanguageAlpha, LanguageBeta."

        with patch("agent_loop.ask_llm_for_decision", return_value={
            "action": "final",
            "answer": draft,
        }):
            answer = ground_final_answer(state, draft)

        self.assertIn("LanguageAlpha", answer)
        self.assertIn("LanguageBeta", answer)
        self.assertIn("does not state the candidate's favorite programming language", answer)
        self.assertNotIn("LanguageAlpha is the candidate's favorite", answer)
        self.assertFalse(state.grounding_verified)

    def test_explicit_resume_favorite_remains_answerable(self):
        resume = """Casey Morgan
SKILLS
Programming: LanguageAlpha, LanguageBeta
Favorite programming language: LanguageBeta
"""
        state = self._favorite_state(resume)
        answer = "LanguageBeta is the candidate's favorite programming language."

        grounded_answer = ground_final_answer(state, answer)

        self.assertEqual(grounded_answer, answer)
        self.assertTrue(state.grounding_verified)


class CandidateNameRoutingTests(unittest.TestCase):
    def assert_name_query_returns_parsed_name(self, resume_text, query):
        parsed_name = parse_resume(resume_text)["candidate_name"]
        self.assertIsNotNone(parsed_name)

        result = search_resume(query, resume_text)

        self.assertIn(f"Candidate name: {parsed_name}", result["results"])
        self.assertNotIn("not_found_terms", result)

    def test_natural_language_candidate_name_queries_use_parsed_identity(self):
        questions = (
            "What is the candidate's name?",
            "What is the candidate's full name?",
            "Who is the candidate?",
            "Can you tell me the candidate's name?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assert_name_query_returns_parsed_name(SYNTHETIC_RESUME, question)

    def test_exact_candidate_name_query_remains_supported_and_generic(self):
        other_resume = SYNTHETIC_RESUME.replace("Alex Rivera", "Casey Morgan")
        for resume_text in (SYNTHETIC_RESUME, other_resume):
            with self.subTest(candidate=parse_resume(resume_text)["candidate_name"]):
                self.assert_name_query_returns_parsed_name(resume_text, "candidate name")

    def test_run_agent_preflights_name_query_without_calling_llm(self):
        parsed_name = parse_resume(SYNTHETIC_RESUME)["candidate_name"]
        with patch("agent_loop.ask_llm_for_decision") as ask_llm:
            outcome = run_agent(
                "What is the candidate's name?",
                SYNTHETIC_RESUME,
            )

        ask_llm.assert_not_called()
        self.assertEqual(outcome["status"], "final")
        self.assertEqual(outcome["iterations"], 1)
        self.assertEqual(outcome["tools_executed"], 1)
        self.assertTrue(outcome["grounding_verified"])
        self.assertIn(parsed_name, outcome["answer"])

    def test_candidate_name_query_fails_closed_when_name_is_absent(self):
        with patch("agent_loop.ask_llm_for_decision") as ask_llm:
            outcome = run_agent(
                "What is the candidate's name?",
                NO_NAME_RESUME,
            )

        ask_llm.assert_not_called()
        self.assertEqual(outcome["tools_executed"], 1)
        self.assertFalse(outcome["grounding_verified"])
        self.assertIn("no evidence", outcome["answer"].lower())

    def test_unrelated_name_question_is_not_routed_to_candidate_identity(self):
        result = search_resume(
            "What does the candidate's name mean for project fit?",
            SYNTHETIC_RESUME,
        )

        self.assertNotIn("Candidate name: Alex Rivera", result["results"])


class OllamaWarmupTests(unittest.TestCase):
    def test_warmup_uses_fixed_minimal_prompt_and_keep_alive(self):
        response = Mock()
        response.json.return_value = {"response": "ready"}
        with patch("agent_loop.requests.post", return_value=response) as post:
            warm_up_ollama()

        args, kwargs = post.call_args
        payload = kwargs["json"]
        self.assertEqual(args[0], OLLAMA_URL)
        self.assertEqual(payload["model"], MODEL_NAME)
        self.assertEqual(payload["prompt"], OLLAMA_WARMUP_PROMPT)
        self.assertNotIn("Alex Rivera", payload["prompt"])
        self.assertNotIn("What responsibilities", payload["prompt"])
        self.assertEqual(payload["options"]["num_predict"], 1)
        self.assertEqual(payload["keep_alive"], OLLAMA_KEEP_ALIVE)
        self.assertEqual(kwargs["timeout"], OLLAMA_WARMUP_TIMEOUT)

    def test_normal_generations_refresh_model_keep_alive(self):
        response = Mock()
        response.json.return_value = {"response": "ready"}
        with patch("agent_loop.requests.post", return_value=response) as post:
            self.assertEqual(call_ollama("synthetic prompt", max_tokens=1), "ready")

        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], OLLAMA_KEEP_ALIVE)

    def test_startup_reports_ready_when_warmup_succeeds(self):
        with patch("api.warm_up_ollama") as warmup:
            with TestClient(api.app) as client:
                health = client.get("/health")

        warmup.assert_called_once_with()
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "model_ready": True})

    def test_startup_continues_safely_when_warmup_fails(self):
        sentinel = "synthetic private-data sentinel"
        with patch("api.warm_up_ollama", side_effect=RuntimeError(sentinel)):
            with self.assertLogs("resume_api", level="WARNING") as captured:
                with TestClient(api.app) as client:
                    health = client.get("/health")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "model_ready": False})
        self.assertNotIn(sentinel, "\n".join(captured.output))


class UploadPrivacyAndSizeTests(unittest.TestCase):
    def setUp(self):
        self.previous_active_resume = api.get_active_resume_text()
        api.set_active_resume_text("")

    def tearDown(self):
        api.set_active_resume_text(self.previous_active_resume)

    def test_metadata_only_upload_size_limit_and_rejected_upload_state(self):
        pdf_bytes = make_synthetic_pdf()
        self.assertLess(len(pdf_bytes), api.MAX_UPLOAD_SIZE_BYTES)

        with patch("api.warm_up_ollama"), TestClient(api.app) as client:
            success = client.post(
                "/upload-resume",
                files={"file": ("synthetic-resume.pdf", pdf_bytes, "application/pdf")},
            )
            self.assertEqual(success.status_code, 200, success.text)
            payload = success.json()
            self.assertEqual(set(payload), {"filename", "pages"})
            self.assertEqual(payload["filename"], "synthetic-resume.pdf")
            self.assertIsInstance(payload["pages"], int)
            self.assertNotIn("raw_text", payload)
            self.assertNotIn("text", payload)

            active_resume = api.get_active_resume_text()
            self.assertTrue(active_resume)

            oversized = client.post(
                "/upload-resume",
                files={
                    "file": (
                        "oversized.pdf",
                        b"x" * (api.MAX_UPLOAD_SIZE_BYTES + 1),
                        "application/pdf",
                    )
                },
            )
            self.assertEqual(oversized.status_code, 413, oversized.text)
            self.assertIn("10 MiB", oversized.json()["detail"])
            self.assertEqual(api.get_active_resume_text(), active_resume)


if __name__ == "__main__":
    unittest.main(verbosity=2)
