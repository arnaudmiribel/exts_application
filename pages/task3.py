import streamlit as st
from .utils import intuition_header, code_header, exercises_header, write_answer, write_instructor_expectation

def write():

    st.title("Task 3: Sample learner question")
    st.markdown("""
    > Write a short answer to this hypothetical learner question. You can assume that the learner is familiar with Python and Scikit-learn, and has high school mathematics knowledge.
    > 
    > **Do we always need to normalize our data? When should we use min-max, or zero mean/unit variance scaling, what's the difference?**
    """)

    st.markdown("""

Hereafter, we refer to scaling as the general term for min-max scaling or normalization or standardization.

This is a very interesting question. Find below some of the reasons why scaling is interesting, and particular situations
where min-max could be more suited than zero mean/unit variance scaling.

### Why scaling?

Imagine we were to use a dataset with heights and incomes as features. These two features are expressed using different
scales, respectively centimeters and CHFs. Height might stand in the interval $[140-200]$ while income will typically
reach thousands, which is at least ten times bigger than height. 

If you build a model like a multi-variate linear regression
based on the features as such, that will intrinsically mean that income will influence the result more. 
Indeed, a slight variation of income won't have the same impact than a slight variation of height, because magnitudes 
are 10 times bigger. And yet, there's no reason to think that income should be a more important predictor 
from the beginning! Hence, we scale the data.


### Which scaling? Min-max or zero mean/unit variance?

Usually, you will want to use zero mean/unit variance scaling, as it is the most natural solution to the points raised above.

Yet, in some cases, using min-max scaling will be more adapted than zero mean/unit variance:
- If your data is very sparse (with many zeros, e.g. [document-term matrices](https://en.wikipedia.org/wiki/Document-term_matrix)), you
will be reluctant to standardize your data, as it would prevent you from exploiting all the computation benefits of having sparse data. 
Min-max scaling, instead, would work fine!
- If your feature's variance is very small, when standardizing, you will divide by the variance so your data will
 get very big.  You end up not controling at all the interval in which lies 
your data, which was the initial goal of scaling. Min-max scaling, instead, would work fine here again!


🏁 All in all, it doesn't hurt to try with **and** without scaling to make sure it improves your model performance. 
Try both, and if it doesn't hurt, better keep it for the reasons mentioned above. Scalers are implemented in 
[`sklearn.preprocessing`](https://scikit-learn.org/stable/modules/preprocessing.html).
""")

    write_instructor_expectation("""
As was asked in the question, I stayed pretty high level and provided a short answer, with an emphasis 
on features different units as it is the most visual in my opinion.

Some other interesting aspects could have been addressed, for example:
- Scaling as a convention: scaling can be seen a standard pre-processing and even sometimes as a requirement for some models
(e.g. radial basis function kernels of SVMs)
- Numeric stability (for gradients and regularization).
- Importance of standardization for dimension reduction techniques like PCA.

This tip is also interesting and could have fed a question:  
💡 At training time, remember to store both $\mu_{train}$ and $\sigma_{train}$.   Indeed, at testing time, 
you will want to use them for standardizing $X_{test}$ (and definitely not use 
$\mu_{test}$ and $\sigma_{test}$). Don't worry, `sklearn` handles that under the hood, but it's good to know.
""")

    if st.button("🎈 That's all for this task. Click to celebrate!"):
        st.balloons()