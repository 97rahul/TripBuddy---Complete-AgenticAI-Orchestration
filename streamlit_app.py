import streamlit as st

from backend import resume_travel_agent, run_travel_agent


st.set_page_config(
    page_title="TripBuddy AI Travel Planner",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ TripBuddy - Your AI Travel Planner")
st.caption("Plan flights, hotels, weather, and a complete trip itinerary in minutes.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "travel_result" not in st.session_state:
    st.session_state.travel_result = None

with st.sidebar:
    st.header("Trip Planner")
    st.write("Describe your trip and let the AI build a practical travel plan.")
    st.markdown(
        """
        Example prompts:
        - 7-day Japan trip from India
        - Weekend trip to Bali under $800
        - 5-day Paris itinerary with flights from Delhi
        """
    )

prompt = st.text_area(
    "Your travel request",
    placeholder="Example: Plan a 5-day Kyoto trip from Mumbai with budget-friendly hotel options.",
    height=180,
)

col1, col2 = st.columns([1, 3])
with col1:
    submit = st.button("Plan My Trip", use_container_width=True)

if submit:
    if not prompt.strip():
        st.warning("Please enter a travel request before planning your trip.")
    else:
        try:
            with st.spinner("Researching flights, hotels, weather, and itinerary..."):
                result = run_travel_agent(prompt, st.session_state.thread_id)
            st.session_state.travel_result = result
            st.session_state.thread_id = result.get("thread_id")
        except Exception as exc:
            print(f"Travel planning failed: {exc}")
            st.error("TripBuddy could not complete this request. Please try again.")

result = st.session_state.travel_result
if result:
    if result.get("requires_approval"):
        st.info(
            result.get("approval_request")
            or "Review the draft itinerary before finalizing your plan."
        )
        st.markdown("## Draft itinerary")
        st.markdown(result.get("itinerary") or result.get("answer") or "No draft generated.")

        with st.form("itinerary_approval"):
            feedback = st.text_area(
                "What should be changed? (Optional if you request changes)",
                placeholder="For example: reduce the daily budget or add more time for museums.",
            )
            approve = st.form_submit_button("Approve itinerary")
            request_changes = st.form_submit_button("Request changes and finalize")

        if approve or request_changes:
            try:
                with st.spinner("Finalizing your travel plan..."):
                    result = resume_travel_agent(
                        result["thread_id"],
                        approved=approve,
                        feedback=feedback,
                    )
                st.session_state.travel_result = result
            except Exception as exc:
                print(f"Travel plan approval failed: {exc}")
                st.error("TripBuddy could not finalize this plan. Please try again.")

    if not result.get("requires_approval"):
        if result.get("guardrail_allowed") is False:
            st.info(result.get("answer") or result.get("guardrail_reason"))
        else:
            st.success("Your travel plan is ready.")
            st.markdown("## Final Travel Plan")
            st.markdown(result.get("answer") or "No response generated.")

    st.markdown("---")

    with st.expander("Flight Details", expanded=False):
        flights = result.get("flight_results") or "No flight results available."
        if isinstance(flights, str):
            st.text(flights)
        else:
            st.json(flights)

    with st.expander("Hotel Suggestions", expanded=False):
        hotels = result.get("hotel_results") or "No hotel results available."
        if isinstance(hotels, str):
            st.text(hotels)
        else:
            st.json(hotels)

    with st.expander("Weather Information", expanded=False):
        weather = result.get("weather_results") or "No weather information available."
        if isinstance(weather, str):
            st.text(weather)
        else:
            st.json(weather)

    with st.expander("Budget Assessment", expanded=False):
        budget = result.get("budget_results") or "No budget assessment available."
        if isinstance(budget, str):
            st.markdown(budget)
        else:
            st.json(budget)

    if not result.get("requires_approval"):
        with st.expander("Day-by-Day Itinerary", expanded=False):
            itinerary = result.get("itinerary") or "No itinerary generated."
            if isinstance(itinerary, str):
                st.markdown(itinerary)
            else:
                st.json(itinerary)

    with st.expander("Agent Info", expanded=False):
        st.write(f"Thread ID: {result.get('thread_id')}")
        st.write(f"LLM Calls: {result.get('llm_calls', 0)}")
        if result.get("selected_agents"):
            st.write("Selected agents:", ", ".join(result["selected_agents"]))

else:
    st.info("Enter a trip request and click the button to generate a travel plan.")
