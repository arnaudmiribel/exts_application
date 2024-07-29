"""Main module for the streamlit app"""

import streamlit as st
import pages.home as home
import pages.task1 as task1
import pages.task2 as task2
import pages.task3 as task3

PAGES = {
    "Home": home,
    "Task 1: Naïves Bayes": task1,
    "Task 2: F1 score": task2,
    "Task 3: Scaling": task3,
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
