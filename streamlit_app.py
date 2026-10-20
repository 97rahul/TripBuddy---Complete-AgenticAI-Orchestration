import streamlit as st

from backend import run_travel_agent


st.set_page_config(
    page_title="TripBuddy AI Travel Planner",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ TripBuddy - Your AI Travel Planner")
st.caption("Plan flights, hotels, and a complete trip itinerary in minutes.")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

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
        with st.spinner("Researching flights, hotels, and itinerary..."):
            result = run_travel_agent(prompt, st.session_state.thread_id)
            st.session_state.thread_id = result.get("thread_id")

        st.success("Trip plan generated.")

        response = result.get("answer") or "No response generated."
        st.markdown("## Final Travel Plan")
        st.markdown(response)

        st.markdown("---")

        with st.expander("Flight Details", expanded=True):
            flights = result.get("flights_results") or "No flight results available."
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

        with st.expander("Day-by-Day Itinerary", expanded=False):
            itinerary = result.get("itinerary") or "No itinerary generated."
            if isinstance(itinerary, str):
                st.markdown(itinerary)
            else:
                st.json(itinerary)

        with st.expander("Agent Info", expanded=False):
            st.write(f"Thread ID: {result.get('thread_id')}")
            st.write(f"LLM Calls: {result.get('llm_calls', 0)}")

else:
    st.info("Enter a trip request and click the button to generate a travel plan.")
