"""SpecGuard Studio: a Streamlit interface for the existing LangGraph workflow.

Keep this file next to main.py. It loads the graph definitions without executing
main.py's command-line loop, so no changes to main.py are required.
"""

import ast
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="SpecGuard / Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { color-scheme: light; }
html, body, [data-testid="stAppViewContainer"] { background: #F3F0E9 !important; color: #242521 !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #E8E4DB !important; }
.block-container { max-width: 1360px !important; padding-top: 1.1rem !important; padding-bottom: 4rem !important; }
* { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.05em !important; color: #242521 !important; }
[data-testid="stMarkdownContainer"] p { line-height: 1.55; }
.sg-top { border-top: 2px solid #272824; border-bottom: 1px solid #B8B6AE; padding: 14px 0 15px; display:flex; align-items:center; justify-content:space-between; gap:12px; }
.sg-brand { font-family:'Space Grotesk',sans-serif; font-size:26px; font-weight:700; letter-spacing:-1.5px; }
.sg-brand span { color:#F04B2E; }
.sg-edition { font: 11px 'DM Mono',monospace; letter-spacing:1.3px; color:#696A62; }
.sg-hero { margin: 30px 0 17px; }
.sg-eyebrow { font: 11px 'DM Mono',monospace; letter-spacing:1.6px; text-transform:uppercase; color:#E14B31; font-weight:500; }
.sg-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(48px,6.7vw,96px); line-height:.98; font-weight:700; letter-spacing:-6px; color:#22231F; margin:14px 0; }
.sg-title i { font-style:normal; color:#E84B30; }
.sg-lead { font-size:17px; max-width:660px; color:#67685F; margin:10px 0 28px; }
.sg-rule { border-top: 1px solid #B8B6AE; margin: 14px 0 23px; }
.sg-label { font: 11px 'DM Mono',monospace; color:#797A72; letter-spacing:1.2px; text-transform:uppercase; }
.sg-section { font:700 25px 'Space Grotesk',sans-serif; letter-spacing:-1px; margin:5px 0 10px; }
.sg-panel { border:1px solid #C8C5BB; padding:18px 20px; background:#FAF8F3; margin:8px 0 16px; }
.sg-index { font: 12px 'DM Mono',monospace; color:#E84B30; }
.sg-small { font-size:13px; color:#6F7068; }
.sg-kicker { font:11px 'DM Mono',monospace; letter-spacing:1px; color:#7A7B73; }
.sg-stat { border-top: 3px solid #252621; padding:12px 0; }
.sg-stat strong { font:700 37px 'Space Grotesk',sans-serif; letter-spacing:-2px; display:block; color:#252621; }
.sg-stat span { font:11px 'DM Mono',monospace; color:#73756C; text-transform:uppercase; }
.sg-steps { display:flex; gap:5px; margin: 14px 0 24px; }
.sg-step { flex:1; padding:12px 10px; border-top:3px solid #C9C6BD; font:11px 'DM Mono',monospace; color:#92938A; }
.sg-step.active { border-color:#E84B30; background:#EAE5DB; color:#252621; }
.sg-step.done { border-color:#252621; color:#252621; }
.sg-step b { display:block; font-size:10px; margin-bottom:6px; }
.sg-note { border-left:3px solid #E84B30; background:#EAE5DB; padding:12px 17px; margin:15px 0; font-size:14px; }
.stButton > button[kind="primary"] { background:#E84B30 !important; border:1px solid #E84B30 !important; color:#FFF !important; border-radius:2px !important; font-family:'Space Grotesk',sans-serif !important; font-weight:700 !important; min-height:46px; }
.stButton > button[kind="secondary"] { background:#F3F0E9 !important; border:1px solid #262721 !important; color:#262721 !important; border-radius:2px !important; min-height:46px; }
.stButton > button:hover { border-color:#E84B30 !important; }
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input { border-radius:2px !important; border:1px solid #BAB8AF !important; background:#FAF9F5 !important; color:#272821 !important; }
[data-testid="stSelectbox"] > div > div { border-radius:2px !important; background:#FAF9F5 !important; }
[data-testid="stMarkdownContainer"] code { background:#E5E0D6; color:#242521; }
.sg-footer { border-top:1px solid #B8B6AE; padding-top:15px; margin-top:40px; font:11px 'DM Mono',monospace; color:#8B8D84; }
@media (max-width:700px) { .sg-title { letter-spacing:-3px; } .sg-edition { display:none; } .sg-step { font-size:9px; padding:9px 4px; } }
</style>
""",
    unsafe_allow_html=True,
)


def load_existing_graph():
    """Load the existing graph without executing main.py's interactive CLI."""
    source = ROOT / "main.py"
    if not source.is_file():
        raise FileNotFoundError("main.py must be in the same folder as specguard_studio.py")

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    selected = []
    for node in tree.body:
        # The first CLI execution statement starts with: result = graph.invoke(...)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "result" for target in targets):
                break
        selected.append(node)

    if not selected:
        raise RuntimeError("Could not find the existing graph definitions")
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"__name__": "specguard_graph", "__file__": str(source)}
    exec(compile(module, str(source), "exec"), namespace)
    if "graph" not in namespace:
        raise RuntimeError("The existing main.py does not define a compiled 'graph'")
    return namespace["graph"]


@st.cache_resource(show_spinner=False)
def get_graph():
    return load_existing_graph()


def reset_session():
    st.session_state.sg_thread = str(uuid.uuid4())
    st.session_state.sg_result = None
    st.session_state.sg_description = ""
    st.session_state.sg_depth = "standard"
    st.session_state.sg_error = ""


if "sg_thread" not in st.session_state:
    reset_session()


def execute_graph(answer=None):
    st.session_state.sg_error = ""
    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": st.session_state.sg_thread}}
        if answer is None:
            state = {
                "project_description": st.session_state.sg_description,
                "clarification_count": 0,
                "needs_clarification": False,
                "status": "new",
                "result": "",
                "identified_requirements": [],
                "missing_information": [],
                "contradictions": [],
                "clarification_question": "",
                "analysis_depth": st.session_state.sg_depth,
                "revision_request": "",
            }
            result = graph.invoke(state, config=config)
        else:
            result = graph.invoke(Command(resume=answer), config=config)
        st.session_state.sg_result = result
    except Exception as exc:
        st.session_state.sg_error = f"{type(exc).__name__}: {exc}"


st.markdown(
    '<div class="sg-top"><div class="sg-brand">SPEC<span>GUARD</span> <span style="color:#252621;font-size:12px;letter-spacing:1px">/ STUDIO</span></div><div class="sg-edition">ENGINEERING WORKSPACE&nbsp;&nbsp; / &nbsp;&nbsp;VOL. 01</div></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([3.3, 1], gap="large")
with left:
    st.markdown(
        '<div class="sg-hero"><div class="sg-eyebrow">01 / FROM AMBIGUITY TO CLARITY</div><div class="sg-title">Make specs.<br><i>Make sense.</i></div><div class="sg-lead">An editorial workspace for turning rough ideas into structured, reviewable software requirements. Less guesswork. More signal.</div></div>',
        unsafe_allow_html=True,
    )
with right:
    st.markdown('<div style="height:43px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sg-stat"><strong>01</strong><span>Describe the idea</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sg-stat"><strong>02</strong><span>Resolve ambiguity</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sg-stat"><strong>03</strong><span>Review the spec</span></div>', unsafe_allow_html=True)

result = st.session_state.sg_result
interrupted = bool(result and "__interrupt__" in result)
approved = bool(result and result.get("status") == "approved" and not interrupted)
review = bool(interrupted and result.get("status") == "spec_generated")
clarifying = bool(interrupted and not review)
current_step = 3 if approved or review else (2 if clarifying else 1)

step_names = ["BRIEF", "CLARIFY", "REVIEW"]
steps = '<div class="sg-steps">'
for idx, name in enumerate(step_names, start=1):
    css = "done" if idx < current_step else ("active" if idx == current_step else "")
    steps += f'<div class="sg-step {css}"><b>0{idx} /</b>{name}</div>'
steps += '</div>'
st.markdown(steps, unsafe_allow_html=True)

if st.session_state.sg_error:
    st.error(st.session_state.sg_error)

if result is None:
    st.markdown('<div class="sg-label">NEW ANALYSIS / 001</div><div class="sg-section">Start with the messy idea.</div>', unsafe_allow_html=True)
    st.markdown('Describe the system in your own words. The agent will identify requirements and ask for essential missing details.')
    description = st.text_area(
        "Project description",
        value=st.session_state.sg_description,
        height=210,
        placeholder="Example: A library management system for students and librarians...",
        key="brief_editor",
    )
    depth = st.selectbox(
        "Analysis depth",
        options=["basic", "standard", "detailed"],
        index=["basic", "standard", "detailed"].index(st.session_state.sg_depth),
        help="Basic: essential details. Standard: balanced. Detailed: more thorough analysis.",
    )
    col_action, col_hint = st.columns([1, 2])
    with col_action:
        if st.button("ANALYZE PROJECT  ↗", type="primary", use_container_width=True, disabled=not description.strip()):
            st.session_state.sg_description = description.strip()
            st.session_state.sg_depth = depth
            with st.spinner("Analyzing requirements..."):
                execute_graph()
            st.rerun()
    with col_hint:
        st.markdown('<div class="sg-note">The agent uses your existing LangGraph workflow. No new requirements are assumed to be confirmed.</div>', unsafe_allow_html=True)

else:
    a, b, c = st.columns(3)
    with a:
        st.markdown(f'<div class="sg-stat"><strong>{len(result.get("identified_requirements", [])):02d}</strong><span>Identified requirements</span></div>', unsafe_allow_html=True)
    with b:
        st.markdown(f'<div class="sg-stat"><strong>{len(result.get("missing_information", [])):02d}</strong><span>Information gaps</span></div>', unsafe_allow_html=True)
    with c:
        st.markdown(f'<div class="sg-stat"><strong>{len(result.get("contradictions", [])):02d}</strong><span>Potential conflicts</span></div>', unsafe_allow_html=True)

    if clarifying:
        question = result["__interrupt__"][0].value
        st.markdown('<div class="sg-label">ANALYST INTERVIEW / IN PROGRESS</div><div class="sg-section">One question at a time.</div>', unsafe_allow_html=True)
        st.markdown('<div class="sg-panel"><div class="sg-index">QUESTION FROM THE ANALYST</div></div>', unsafe_allow_html=True)
        st.info(str(question), icon="✳️")
        with st.form("clarification_form", clear_on_submit=True):
            answer = st.text_area("Your answer", height=120, placeholder="Add the missing detail...")
            submitted = st.form_submit_button("SEND ANSWER  ↗", type="primary", use_container_width=True)
            if submitted and answer.strip():
                with st.spinner("Re-evaluating the requirements..."):
                    execute_graph(answer.strip())
                st.rerun()
        if st.button("SKIP REMAINING QUESTIONS", use_container_width=False):
            with st.spinner("Preparing specification..."):
                execute_graph("done")
            st.rerun()

    if review or approved:
        st.markdown('<div class="sg-label">SPECIFICATION / HUMAN REVIEW</div><div class="sg-section">A document worth reviewing.</div>', unsafe_allow_html=True)
        if approved:
            st.success("Specification approved. This workflow is complete.")
        else:
            st.markdown('<div class="sg-note">The specification is a draft. Approve it or request precise changes before finalizing.</div>', unsafe_allow_html=True)
        spec = result.get("result", "")
        st.markdown('<div class="sg-rule"></div>', unsafe_allow_html=True)
        st.markdown(spec)
        st.markdown('<div class="sg-rule"></div>', unsafe_allow_html=True)
        st.download_button(
            "DOWNLOAD SPECIFICATION  ↓",
            data=spec,
            file_name="SpecGuard_Specification.md",
            mime="text/markdown",
            use_container_width=False,
        )
        if review:
            st.markdown('### Human review')
            revision = st.text_area("Requested changes", placeholder="Example: Add negative test cases for unavailable books.", height=110)
            approve_col, revise_col = st.columns(2)
            with approve_col:
                if st.button("APPROVE SPECIFICATION  ✓", type="primary", use_container_width=True):
                    with st.spinner("Finalizing..."):
                        execute_graph("approve")
                    st.rerun()
            with revise_col:
                if st.button("REQUEST REVISION  ↗", use_container_width=True, disabled=not revision.strip()):
                    with st.spinner("Revising specification..."):
                        execute_graph(revision.strip())
                    st.rerun()

    with st.expander("Analysis details", expanded=False):
        st.markdown("**Identified requirements**")
        for item in result.get("identified_requirements", []):
            st.markdown(f"- {item}")
        st.markdown("**Missing information**")
        for item in result.get("missing_information", []):
            st.markdown(f"- {item}")
        st.markdown("**Contradictions**")
        for item in result.get("contradictions", []):
            st.markdown(f"- {item}")

    st.markdown('<div class="sg-rule"></div>', unsafe_allow_html=True)
    if st.button("＋  NEW PROJECT / RESET WORKSPACE"):
        reset_session()
        st.rerun()

st.markdown('<div class="sg-footer">SPECGuard AI &nbsp;•&nbsp; LangGraph-powered &nbsp;•&nbsp; Human-reviewed &nbsp;•&nbsp; Built for clarity, not decoration.</div>', unsafe_allow_html=True)
