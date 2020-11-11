import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit as st

intuition_header = "🧠 Intuition"
code_header = "💻  Demo"
exercises_header = "✍️  Exercises"
success_message = "🎈😎🤩!"
failure_message = "Try again..."


def plot_f1_score():
    N = 100
    X, Y = np.linspace(0.1, 1, N), np.linspace(0.1, 1, N)
    Z = [[(2 * x * y) / (x + y) for x in X] for y in Y]

    fig = go.Figure(
        data=[
            go.Contour(
                z=Z,
                x=X,
                y=Y,
                contours=dict(
                    coloring="heatmap",
                    showlabels=True,  # show labels on contours
                    labelfont=dict(size=12, color="white",),  # label font properties
                ),
            )
        ],
    )

    fig.update_layout(
        title=dict(text="Contours of F1 score based on precision and recall", x=0.5),
        font=dict(family="IBM Plex Sans", color="black"),
        autosize=False,
        width=300,
        height=300,
        xaxis_title="Precision",
        yaxis_title="Recall",
    )
    st.plotly_chart(fig)


def space():
    return st.markdown("<br><br>", unsafe_allow_html=True)


def write_answer(text: str):
    return st.success(
        f"""**Answer:** 
    {text}"""
    )


def write_instructor_expectation(text: str):
    return st.info(
        f"""** ⓘ Instructor expectation:** 
    {text}"""
    )


if __name__ == "__main__":
    pass
