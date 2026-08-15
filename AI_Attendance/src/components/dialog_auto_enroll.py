import time

import streamlit as st

from src.database.config import supabase
from src.database.db import enroll_student_to_subject


@st.dialog("Quick Enrollment")
def auto_enroll_dialog(subject_code):
    student_data = st.session_state.get("student_data")
    if not student_data:
        st.error("Student information is not available. Please log in again.")
        return

    student_id = student_data.get("student_id")
    if not student_id:
        st.error("Student ID is missing. Please log in again.")
        return

    res = (
        supabase.table("subjects")
        .select("subject_id, name")
        .eq("subject_code", subject_code)
        .execute()
    )

    if not res.data:
        st.error("Subject Code not found!")
        if st.button("Close"):
            st.query_params.clear()
            st.rerun()
        return

    subject = res.data[0]

    check = (
        supabase.table("subject_students")
        .select("*")
        .eq("subject_id", subject["subject_id"])
        .eq("student_id", student_id)
        .execute()
    )

    if check.data:
        st.info("You're already enrolled!")
        if st.button("Got it!"):
            st.query_params.clear()
            st.rerun()
        return

    subject_name = subject.get("name", "this subject")
    st.markdown(f"Would you like to enroll in **{subject_name}**?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("No thanks"):
            st.query_params.clear()
            st.rerun()

    with col2:
        if st.button("Yes enroll now!", type="primary", width="stretch"):
            enroll_student_to_subject(student_id, subject["subject_id"])
            st.success("Joined successfully!")
            st.query_params.clear()
            time.sleep(2)
            st.rerun()
