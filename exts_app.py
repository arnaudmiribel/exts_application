"""Main module for the streamlit app"""

import streamlit as st
import pages.home
import pages.task1
import pages.task2
import pages.task3

PAGES = {
    "Home": pages.home,
    "Task 1: Naïves Bayes": pages.task1,
    "Task 2: F1 score": pages.task2,
    "Task 3: Scaling": pages.task3,
}


def main():
    st.sidebar.title("My application for EXTS' instructor position.")
    selection = st.sidebar.radio("Table of contents:", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.write()
    
    st.sidebar.markdown("""
    © [Arnaud Miribel](https://arnaudmiribel.github.io)
    """)


if __name__ == "__main__":
    main()