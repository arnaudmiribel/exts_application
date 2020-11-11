import streamlit as st
import pandas as pd
import numpy as np
from .utils import intuition_header, code_header, exercises_header, write_answer, write_instructor_expectation

def write():

    st.title("Task 1: Creating a demo unit with exercises")

    st.markdown("> In 500-1’000 words, explain to someone the main ideas behind the Naive Bayes classifier. Assume that your audience has some some basic programming and high school level mathematics knowledge, but are not computer scientists, programmers, mathematicians or statisticians. Your material can include text, images and/or code. Separately, please develop at least two exercises for the learner to complete to reinforce this topic.")

    st.header(intuition_header)
    
    st.markdown("""
    Today, we are going to learn how to build a Naïve Bayes classifier to help decide whether John will go skiing or 
    lifting weights at the gym using weather and environment data. John is a very inspiring human being, truly 
    reliable when it comes to choosing a proper sport activity. Our dataset contains records from his activities 
    during last winter holidays: 
    """)
    dataset = pd.DataFrame([
        ["☀️", "🏔️", "🏂"],
        ["☀️", "🏔️", "🏂"],
        ["❄️", "🏙️", "🏋️‍♂️"],
        ["☀️", "🏔️", "🏂"],
        ["❄️", "🏔️", "🏋️‍♂️"],
        ["☀️", "🏙️", "🏋️‍♂️"],
        ["❄️", "🏙️", "🏋️‍♂️"],
        ["❄️", "🏔️", "🏂"],
        ["☀️", "🏔️", "🏋️‍♂️"],
        ["☀️", "🏔️", "🏂"]],
        columns=["weather", "environment", "John's activity"],
        index=["2019-12-20", "2019-12-21", "2019-12-22", "2019-12-23", "2019-12-24", "2019-12-25", "2019-12-26", "2019-12-27", "2019-12-28", "2019-12-29"]
    )

    st.table(dataset)

    st.markdown("""
    ## Problem statement
    Weather can be sunny ( ☀️) or snowy ( ❄️), while John can be downtown ( 🏙️) or in the mountains ( 🏔️). Depending on these 
    factors, John will either go snowboarding ( 🏂) or lifting ( 🏋️‍♂️).


    As you may notice, it's not straightforward to guess if a sunny day will make John go skiing or lifting.
    Also, you will find samples of data where John has been to the gym while being in the mountains or downtown...
    Supposing we get the weather and environment data for a future day, our question is the following:
    """)

    st.warning("""
How can we guess the most likely activity John will do?
""")

    st.markdown("""
    ## Data pre-processing
    We start by creating matrices based on our data, the usual $X$ (features) and $y$ (classes). 
    For that, we have to encode categorical variables (here, emojis) into numeric values.

    We assign ☀️, 🏔️ and 🏂 to ones. And we assign ❄️, 🏙️ and 🏋️‍♂️ to zeros.  Our data becomes:
    """)


    dataset = pd.DataFrame([
        [1, 1, 1],
        [1, 1 ,1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
        [0, 1, 1],
        [1, 1, 0],
        [1, 1, 1]],
        columns=["weather", "environment", "John's activity"],
        index=["2019-12-20", "2019-12-21", "2019-12-22", "2019-12-23", "2019-12-24", "2019-12-25", "2019-12-26", "2019-12-27", "2019-12-28", "2019-12-29"]
    )

    st.table(dataset)

    st.markdown("""
    From which we can extract both $X$ and $y$:
    """)

    st.latex(r"""
    \small
    X = 
    \begin{bmatrix}
    1 & 1\\
    1 & 1\\
    0 & 0\\
    1 & 1\\
    0 & 1\\
    1 & 0\\
    0 & 0\\
    0 & 1\\
    1 & 1\\
    1 & 1\\
    \end{bmatrix}
    \hspace{4em}
    y = 
    \begin{bmatrix}
    1\\
    1\\
    0\\
    1\\
    0\\
    0\\
    0\\
    1\\
    0\\
    1\\
    \end{bmatrix}
    """)

    st.markdown("""## Probability recap
In this section, we will highlight differences between class and conditional probabilities.

### Class probabilities

Class probabilities typically answer a question like: how likely is it that John will go snowboarding or lifting? 
We can compute these class probabilities using:
""")

    st.latex(r"""
    \mathbb{P}(y=0) = \frac{\Omega_{y=0}}{\Omega_{y=0} + \Omega_{y=1}} = \frac{\colorbox{yellow}{5}}{\colorbox{yellow}{5} + \colorbox{cyan}{5}} = 50 \%
    """)
    st.latex(r"""
    \mathbb{P}(y=1) = \frac{\Omega_{y=1}}{\Omega_{y=0} + \Omega_{y=1}} = 1 - \mathbb{P}(y=0) = 50 \%
    """)

    st.markdown("""
    where:
    - $\Omega_{y=i}$ counts the number of occurrences of the event $y=i$ in our dataset  
    - $\mathbb{P(y=i)}$ is the probability of event ${y=i}$ 

    You can actually check visually:
    """)

    st.latex(r"""
    \tiny
    y = 
    \begin{bmatrix}
    1\\
    1\\
    \colorbox{yellow}{0}\\
    1\\
    \colorbox{yellow}{0}\\
    \colorbox{yellow}{0}\\
    \colorbox{yellow}{0}\\
    1\\
    \colorbox{yellow}{0}\\
    1\\
    \end{bmatrix}
    \hspace{3em}
    y = 
    \begin{bmatrix}
    \colorbox{cyan}{1}\\
    \colorbox{cyan}{1}\\
    0\\
    \colorbox{cyan}{1}\\
    0\\
    0\\
    0\\
    \colorbox{cyan}{1}\\
    0\\
    \colorbox{cyan}{1}\\
    \end{bmatrix}
    """)

    st.markdown("""
    So, based on our dataset, John will go snowboarding or lifting with an equal probability of 50%.
    """)

    st.markdown("""
    ### Conditional probabilities

    Now, we look at conditional probabilities based on available weather ($w$) and environment ($e$) features we have. 
    Let's start with weather features (1st column). The probability that the weather is snowy **given that** John went lifting is,
    using chain rule [[remember!]](https://en.wikipedia.org/wiki/Chain_rule_(probability)):
    """)

    st.latex(r"""
    \mathbb{P}(w=0|y=0) = \frac{\mathbb{P}(w=0 \cap y=0)}{\mathbb{P}(y=0)} = \frac{\Omega_{w=0 \cap y=0}}{\Omega_{y=0}} 
    = \frac{\colorbox{cyan}{3}}{\colorbox{yellow}{5}} = 60 \%
    """)

    st.latex(r"""
    \tiny
    X = 
    \begin{bmatrix}
    1 & 1\\
    1 & 1\\
    \colorbox{cyan}{0} & 0\\
    1 & 1\\
    \colorbox{cyan}{0} & 1\\
    1 & 0\\
    \colorbox{cyan}{0} & 0\\
    0 & 1\\
    1 & 1\\
    1 & 1\\
    \end{bmatrix}
    \hspace{4em}
    y = 
    \begin{bmatrix}
    1\\
    1\\
    \colorbox{yellow}{0}\\
    1\\
    \colorbox{yellow}{0}\\
    \colorbox{yellow}{0}\\
    \colorbox{yellow}{0}\\
    1\\
    \colorbox{yellow}{0}\\
    1\\
    \end{bmatrix}
    """)

    st.markdown("""
    Indeed, looking at our $X$ and $y$ matrices, this probability means counting blue events, dividing by yellow events!
    """)

    st.markdown("""
    Given that John's lifting, what's the probability of the weather to be sunny?
    """)

    st.latex(r"""
    \mathbb{P}(w=1|y=0) = \frac{\mathbb{P}(w=1 \cap y=0)}{\mathbb{P}(y=0)} = \frac{\Omega_{w=1 \cap y=0}}{\Omega_{y=0}} 
    = \frac{\colorbox{cyan}{2}}{\colorbox{yellow}{5}} = 40 \%
    """)

    st.latex(r"""
    \tiny
    X = 
    \begin{bmatrix}
    1 & 1\\
    1 & 1\\
    0 & 0\\
    1 & 1\\
    0 & 1\\
    \colorbox{cyan}{1} & 0\\
    0 & 0\\
    0 & 1\\
    \colorbox{cyan}{1} & 1\\
    1 & 1\\
    \end{bmatrix}
    \hspace{4em}
    y = 
    \begin{bmatrix}
    1\\
    1\\
    \colorbox{yellow}{0}\\
    1\\
    \colorbox{yellow}{0}\\
    \colorbox{yellow}{0}\\
    \colorbox{yellow}{0}\\
    1\\
    \colorbox{yellow}{0}\\
    1\\
    \end{bmatrix}
    """)

    st.markdown("""
    So, given that John went lifting, the day was rather snowy than sunny: makes sense... John maybe
    prefers doing outdoor activities when it's sunny, while lifting can be done on snowy days!
    We can compute the remaining conditional probabilities for weather features:
    """)

    st.latex(r"""
    \mathbb{P}(w=0|y=1) = \frac{1}{5} = 20 \% \hspace{4em} \mathbb{P}(w=1|y=1) = \frac{4}{5} = 80 \%
    """)

    st.markdown("""
    And we do this again, looking this time at environment ($e$) features:
    """)

    st.latex(r"""
    \mathbb{P}(e=0|y=0) = \frac{3}{5} = 60 \% \hspace{4em} \mathbb{P}(w=1|y=1) = \frac{2}{5} = 40 \%
    """)

    st.latex(r"""
    \mathbb{P}(e=0|y=1) = \frac{0}{5} = 0 \% \hspace{4em} \mathbb{P}(e=1|y=1) = \frac{5}{5} = 100 \%
    """)

    st.markdown(""" ## Naïve Bayes model""")

    st.markdown("""
    ### First, remember Bayes' theorem!
    As you may have guessed from the model name, we will use Bayes' theorem! It is a simple mathematical 
    formula used for calculating conditional probabilities:
    """)

    st.latex(r"""
    \mathbb{P}(A|B) = \frac{\mathbb{P}(B|A) \cdot \mathbb{P}(A)}{\mathbb{P}(B)} 
    """)

    st.markdown("""
    ### A probability distribution
    
    We would like to define John's activity probability distribution based on a future sample's weather ($w$) and environment ($e$).
    That is, we want to establish a formula that expresses how to calculate $$\mathbb{P}(y|w,e)$$. In fact, using Bayes' theorem:
    """)

    st.latex(r"""
    \mathbb{P}(y|w,e) = \frac{\mathbb{P}(y) \cdot \mathbb{P}(w,e|y)}{\mathbb{P}(w,e)}
    """)

    st.markdown("""
    That's the moment when we introduce and apply the **Naïve hypothesis**: we suppose that weather and environment are independent variables. 
    Here, that's a reasonable assumption, stating that John's environment (mountain or downtown) does not depend 
    on the weather (and vice versa).
    Hence, that means their joint distribution equals the product of their distribution:
    """)

    st.latex(r"""
    \mathbb{P}(w,e) = \mathbb{P}(w) \cdot \mathbb{P}(e) \hspace{2em} \implies  \hspace{2em} \mathbb{P}(w,e|y) = \mathbb{P}(w|y) \cdot \mathbb{P}(e|y)
    """)
    st.markdown("""
    We can thus further develop our expression of $$\mathbb{P}(y|w,e)$$:
    """)

    st.latex(r"""
    \mathbb{P}(y|w,e) = \frac{\mathbb{P}(y) \cdot \mathbb{P}(w|y) \cdot \mathbb{P}(e|y)}{\mathbb{P}(w,e)}
    """)

    st.markdown("""### From distribution to classification

If we were to evaluate a new setting, an usual way to turn this distribution into a classifier is to 
look for the outcome $y$ that maximizes the above probability distribution. This methodology is called
Maximum A Posteriori (MAP). We want to find $y$ such that the expression above is maximized. 
That's the exact role of $\operatorname{argmax}$:
    """)

    st.latex(r"""\begin{aligned}
    \underset{y}{\operatorname{argmax}} \hspace{0.5em} \mathbb{P}(y|w,e) &= \underset{y}{\operatorname{argmax}} \hspace{0.5em} \frac{\mathbb{P}(y) \cdot \mathbb{P}(w|y) \cdot \mathbb{P}(e|y)}{\mathbb{P}(w,e)} \\ 
      &= \underset{y}{\operatorname{argmax}} \hspace{0.5em} \mathbb{P}(y) \cdot \mathbb{P}(w|y) \cdot \mathbb{P}(e|y) \\ 
    \end{aligned}
    """)

    st.markdown("""
    In the expression, we got rid of the denominator because it does not depend on $y$, hence could not impact $\operatorname{argmax}$.
    We're ready to predict now!

    ### Prediction time

    According to our model, what's the most probable activity John will do, given that it's a sunny day ($w=1$) and John in the mountains ($e=1$)? 
    We look, for both possible outcomes, when $y$ maximizes the probability distribution.
    """)

    st.latex(r"""
    \mathbb{P}(y=0) \cdot \mathbb{P}(w=1|y=0) \cdot \mathbb{P}(e=1|y=0) = 0.5 \cdot 0.4 \cdot 0.4 = 0.08
    """)

    st.latex(r"""
    \mathbb{P}(y=1) \cdot \mathbb{P}(w=1|y=1) \cdot \mathbb{P}(e=1|y=1) = 0.5 \cdot 0.8 \cdot 1.0 = 0.4
    """)

    st.markdown("""
    So, because the second probability is higher, our Naïve Bayes classifier's prediction is...
    """)

    st.latex(r"""
    \underset{y}{\operatorname{argmax}} \hspace{0.5em} \mathbb{P}(y) \cdot \mathbb{P}(w=1|y) \cdot \mathbb{P}(e=1|y) = 1
    """)

    st.markdown("""
    ... snowboarding! 🏂🏂🏂
    """)

    st.markdown(""" ### Play with the model
    """)

    st.write("Choose your inputs, look at predictions: do you agree?")
    weather_input = st.selectbox("Weather: ", ["❄️", "☀️"])
    environment_input = st.selectbox("Environment: ", ["🏔️", "🏙️"])
    category_mapper = {"☀️":1, "❄️":0, "🏔️":1, "🏙️": 0, "🏂": 1, "🏋️‍♂️": 0}
    weather_binary, environment_binary = category_mapper[weather_input], category_mapper[environment_input]
    conditional_w = np.array([
        [0.6, 0.2],
        [0.4, 0.8]]
    )
    conditional_e = np.array([
        [0.6, 0.],
        [0.4, 1.]]
    )
    probability_0 = 0.5 * conditional_w[weather_binary, 0] * conditional_e[environment_binary, 0]
    probability_1 = 0.5 * conditional_w[weather_binary, 1] * conditional_e[environment_binary, 1]
    st.markdown(f"Probability of John going to lift: {round(probability_0, 3)}")
    st.markdown(f"Probability of John going to snowboard: {round(probability_1, 2)}")

    if probability_1 > probability_0:
        st.write("**Prediction:** John will go snowboarding! 🏂🏂🏂")
    else:
        st.write("**Prediction:** John will go lifting! 🏋️‍♂️🏋️‍♂️🏋️‍♂️")

    st.header(exercises_header)

    st.markdown("""
$1.1)$ What would the final expression of our probability distribution be if there were two supplementary 
features? Features are:
- John's mood ($m$) being happy ($m=1$) or unhappy ($m=0$)
- John's busy ($b$) with work ($b=1$) or not busy ($b=0$) 
""")

    write_answer(r"""
Using the naïve hypothesis, we still suppose the features are independent. The distribution would be:

$$
\begin{gathered}
\mathbb{P}(y|w,e,m,b) = \frac{\mathbb{P}(y) \cdot \mathbb{P}(w|y) \cdot \mathbb{P}(e|y) \cdot \mathbb{P}(m|y) \cdot \mathbb{P}(b|y)}{\mathbb{P}(w,e,m,b)}
\end{gathered}
$$

Hence the classifier would be:

$$
\begin{gathered}
\underset{y}{\operatorname{argmax}} \hspace{0.5em} \mathbb{P}(y) \cdot \mathbb{P}(w|y) \cdot \mathbb{P}(e|y) \cdot \mathbb{P}(m|y) \cdot \mathbb{P}(b|y)
\end{gathered}
$$

""")

    write_instructor_expectation("""
This is a straightforward question to help the student digest and extrapolate the freshly
learnt concept.
""")

    st.markdown("""
$1.2)$ What is a major weakness of a Naïve Bayes classifier? 
""")

    write_answer("""
Concretely, the naïve hypothesis will prevent you from learning interactions between features. For instance,
a Naïve Bayes classifier won't be able to learn that although you love chocolate and pasta, you won't 
love chocolate and pasta together... So you better not choose it for predicting great recipes! ~~Maybe that's
how people came up with pizza & pineapple by the way.~~.

Other acceptable answers include mentioning the "zero frequency" problem, which is that the model will 
assign zero probabilities for features that were never seen in the training set.
""")

    write_instructor_expectation("""
This question is important to step back and understand what are the limits of what we just learnt.
""")

    st.markdown("""
$1.3)$ Imagine our problem is now text classification. In that case, inputs are text documents,
and output classes can be topics addressed in the text documents (e.g. "sports", "politics"). What features can you think
of to help modeling this problem? Would a Naïve Bayes classifier be suited to this problem?
""")

    write_answer("""
The typical features for text classification could be word presence or absence in a document. In fact, from the set of documents, we
would extract all unique terms (the so-called vocabulary). Vocabulary sizes can reach hundreds or thousands. 
And thus our features would be the presence or absence of these words. 
[[Read more on Document-Term matrices]](https://en.wikipedia.org/wiki/Document-term_matrix).

We talked about Naïve Bayes classifiers weaknesses, but their strength is to be extremely fast to build and 
they scale well with many (hundreds of) features! So text classification, where hundreds of features can be necessary,
is a very successful application of Naïve Bayes classifiers.
    """)

    write_instructor_expectation("""
Building upon question $1.2)$, I thought this question was a nice final question in order to let the student
be creative about a new application and understand how / where Naïve Bayes can be a good candidate.
""")

    if st.button("🎈 That's all for this task. Click to celebrate!"):
        st.balloons()
