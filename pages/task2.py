import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from .utils import intuition_header, code_header, exercises_header, plot_f1_score, write_answer, write_instructor_expectation

def write():

    st.title("Task 2: Scikit-learn demo unit")

    st.markdown("> *Write a 500-1’000 word tutorial about the F1 score and how to use it in Scikit-learn. You can choose the data and model. Please assume that the target audience is familiar with Python and the sklearn library: estimators API (for example, fit and score), related tools (including pipelines), common classification models (such as logistic regressions), but not yet familiar with the precision and recall metrics. Your demo unit should include a minimum of one exercise/example.*")

    st.header(intuition_header)
    
    st.markdown("""
In this section, we will gain intuition on a popular measure to evaluate a binary classifier called the F1 score.
But before doing so, we will address a few concepts tighly related to the F1 score. These concepts are 
confusion matrices, true/false positive/negatives, precision, recall and accuracy.  
""")

    st.markdown("""
**Confusion matrix**  
In a model's confusion matrix, the row index corresponds to the predicted class, while 
the column index corresponds to the true class, such that looking at the matrix enables to quickly get a grasp 
of classes for which your model performs well or poorly. Indeed, looking at diagonal elements give
the number of points for which the predicted class is equal to the true class, while off-diagonal elements 
are those that are missed by the classifier. 
    
The higher the diagonal values of the confusion matrix the better, indicating many 
correct predictions.
""")

    st.latex(r"""
    \begin{bmatrix}
    8 & 1 & 1\\
    3 & 7 & 0\\
    2 & 2 & 6
    \end{bmatrix}
    """)

    st.markdown("""
**TP, TN, FP, FN**  
In the special case of binary classification, the confusion matrix is a $(2, 2)$ matrix which can be directly 
mapped to entities called true positives (TP), true negatives (TN), false positives (FP), false negatives (FN).
True positives are the elements you correctly classified as positives, while false positives are the elements
you thought were positives, but in fact are negatives. And conversely.
""")

    st.latex(r"""
    \begin{bmatrix}
    TP & FN\\
    FP & TN
    \end{bmatrix}
    =
    \begin{bmatrix}
    8 & 2\\
    3 & 7
    \end{bmatrix}

    """)

    st.markdown("""
**Precision and recall**  
Based on TP, TN, FP and FN, we introduce precision and recall as:
""")

    st.latex(r"""
    P = \frac{TP}{TP + FP}
    """)

    st.latex(r"""
    R = \frac{TP}{TP + FN}
    """)

    st.markdown("""
    As a reminder, we also define below the accuracy $A$:
    """)

    st.latex(r"""
    A = \frac{TP + TN}{TP + TN + FP + FN}
    """)

    st.markdown("""
These measures answer different and complementary questions. Imagine our binary classifier is 
predicting whether a patient is sick (positive) or not (negative), then an intuition to precision,
recall and accuracy can be obtained from these questions:
- When my model says the patient is sick: is he really sick? i.e. is my model **precise**?
- Did my model manage to retrieve all the patients that were sick? i.e. is the **recall** good?
- When my model says something, is it right? i.e. is my **accuracy** good?

**F1 score**   
A popular way to combine precision and recall is by taking their harmonic mean.
The resulting expression is that of the **F1 score**. To express it, let's start by 
recalling the definition of the harmonic mean $H$ for positive numbers $x_1, x_2, ... x_n$:
""")

    st.latex(r"""H(x_1, ..., x_n) = \frac{2}{ \frac{1}{x_1} + \frac{1}{x_2} + ... + \frac{1}{x_n} }""")

    st.markdown("""
    Applying this to precision $p$ and recall $r$:
    """)

    st.latex(r"""
    H(p, r) = \frac{2}{ \frac{1}{p} + \frac{1}{r}} 
    """)

    st.markdown("""
    Which can alternatively be expressed as:
    """)

    st.latex(r"""
    H(p, r) = \frac{2}{ \frac{1}{p} + \frac{1}{r}} 
    = \frac{2 \cdot p \cdot r}{(\frac{1}{p} + \frac{1}{r}) \cdot p \cdot r}
    = \frac{2 \cdot p \cdot r}{\frac{p \cdot r}{p} + \frac{p \cdot r}{r}}
    = \frac{2 \cdot p \cdot r}{p + r}
    """)

    st.markdown("""
Finally, find below the usual definition of **F1 score** (denoted as $F_1$):
""")

    st.latex(r"""F_1 = 2 \cdot \frac{p \cdot r}{p+r}""")

    st.markdown("""
Using the visualisation below, you can get an intuition on the potential values of precision and recall
for a given F1 score. Follow the contours!
""")

    plot_f1_score()

    st.markdown("""
Now you know what is F1 score and how to obtain it.  
Let's use it in a real project through `sklearn`!
""")

    st.header(code_header)

    st.markdown("""
In this section, we will see how to use F1 score in practice.
As you are already familiar with the library `sklearn`, we will train a model using `sklearn.Pipelines`
and see how to evaluate it using F1 score. Our use-case is distinguishing men from women based on samples
from heights and weights.

Let's start by collecting data and looking at it
```python
import pandas as pd
import numpy as np

np.random.seed(0)
dataset = pd.read_csv("https://raw.githubusercontent.com/Dataweekends/zero_to_deep_learning_video/master/data/weight-height.csv")
dataset["Weight"] = dataset["Weight"].apply(lambda w: w / 2.205)  # convert pounds to kgs
dataset["Height"] = dataset["Height"].apply(lambda h: h * 2.54)  # convert inches to cms 
dataset.sample(frac=1)  # shuffle
dataset.head()
```
""")

    np.random.seed(0)
    dataset = pd.read_csv("https://raw.githubusercontent.com/Dataweekends/zero_to_deep_learning_video/master/data/weight-height.csv")
    dataset["Weight"] = dataset["Weight"].apply(lambda w: w / 2.205)  # convert pounds to kgs
    dataset["Height"] = dataset["Height"].apply(lambda h: h * 2.54)  # convert inches to cms 
    dataset.sample(frac=1)
    st.write(dataset.head())

    st.markdown("""
How many men and women do we have in this dataset?

```python
dataset["Gender"].value_counts()
```
""")

    st.write(dataset["Gender"].value_counts())

    st.markdown("""
Let's visualise a scatter plot of our data: classes look separable but there's a confusing zone where both men and women 
can belong.
""")
    fig = px.scatter(dataset, x="Weight", y="Height", color="Gender", opacity=0.5)
    st.plotly_chart(fig)

    st.markdown("""
Now, we pre-process our data by standardizing our data, splitting into train and validation sets.
After applying a standard scaler and fitting a Gaussian Naive Bayes classifier, we output precision,
recall and F1 score. As you may notice, `sklearn` comes with all scores already implemented!
""")

    st.code("""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Dummy encode the target variable ("Gender") and create X, y numpy arrays
dataset["Gender"] = dataset["Gender"].apply(lambda gender: int(gender == "Male"))
X = dataset[["Height", "Weight"]].to_numpy()
y = dataset["Gender"].to_numpy()

# Make a train/test split using 30% test size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Fit the model to the train data and predict using pipelined scaling and classifier.
classifier = make_pipeline(StandardScaler(), GaussianNB())
classifier.fit(X_train, y_train)
y_prediction = classifier.predict(X_test)

print(f"F1 score: {round(metrics.f1_score(y_test, y_prediction), 3)}")
print(f"precision: {round(metrics.precision_score(y_test, y_prediction), 3)}")
print(f"recall: {round(metrics.recall_score(y_test, y_prediction), 3)}")
""")

    # Dummy encode the target variable ("Gender") and create X, y numpy arrays
    dataset["Gender"] = dataset["Gender"].apply(lambda gender: int(gender == "Male"))
    X = dataset[["Height", "Weight"]].to_numpy()
    y = dataset["Gender"].to_numpy()

    # Make a train/test split using 30% test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Fit the model to the training data and predict using pipelined scaling and classifier.
    classifier = make_pipeline(StandardScaler(), GaussianNB())
    classifier.fit(X_train, y_train)
    y_prediction = classifier.predict(X_test)

    st.write(f"F1 score: {round(metrics.f1_score(y_test, y_prediction), 3)}")
    st.write(f"precision: {round(metrics.precision_score(y_test, y_prediction), 3)}")
    st.write(f"recall: {round(metrics.recall_score(y_test, y_prediction), 3)}")

    classifier2 = GaussianNB()
    classifier2.fit(X_train, y_train)
    y_prediction2 = classifier2.predict(X_test)

    st.header(exercises_header)


    st.markdown("""
$2.1)$ In the following python code snippet, we define a function to return an F1 score based on inputs `precision`
and `recall`. What do you think of it?
""")

    st.code("""
def f1_score(precision: float, recall: float):
    numerator = precision * recall
    denominator = precision + recall
    return numerator / denominator
""")

    write_answer("""
The function is wrong: it forgets a `2` in the numerator!
Also, it is not robust. It must handle extreme cases e.g. when precision or recall equals 0. 
As for now, there would be a division by **zero** error.
Adding a `if` statement could prevent you from these problems:

```python
def f1_score(precision: float, recall: float):
    if precision == 0 or recall == 0:
        return 0  # do you agree?
    numerator = 2 * precision * recall
    denominator = precision + recall
    return numerator / denominator
```

Do you think of other ways to robustify the function?
""")

    write_instructor_expectation("""
This question is a simple starter to help the student confirm his freshly learnt F1 score.
It is intended to be very easy, but I like the idea of raising awareness on both 
**theory** (a `2` is missing) and **practice** (division by zero) at the same time.
Also, asking for further ways to robustify the function pushes
the student to be a good programmer and check his code is outlier-proof. I would, for instance,
expect the student to add `assert` statements e.g. for checking positiveness of precision and recall.
""")

    st.markdown("""
$2.2)$ Do you agree with the following statement?

*"F1 score is a way to take both precision and recall
into account, making it an ideal performance measure for any binary classification."*
""")

    write_answer("""
This is not always true in practice. One should always be very careful about the performance
measure and choose it carefully. 

Indeed, different applications will have different requirements: when predicting for a disease, 
you may afford false positives (i.e. telling a patient he
is sick while he isn't) but not false negatives (i.e. telling a sick patient he isn't sick). 
Obtaining an F1 score of 0.95 (which sounds reasonable) means the recall could be as low as ~0.91, hence the false negative 
rate could be higher than 8 percent. That means your model would actually fail to find the 
disease once every 13 patients: can you afford that?

Depending on the application of your model, you may want to weight precision more, or recall more. An interesting
direction for this is Fβ score 
(described [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html))
""")

    write_instructor_expectation("""
I think this question is essential as it challenges the critical thinking of the student. 
Choosing an evaluation method is a very important part of a data science journey; and doing 
it wrong may have dramatic consequences.
""")


    st.markdown("""
$2.3)$ How could you take advantage of F1 score when evaluating a classifier that predicts one among more than two classes?
""")

    write_answer("""
You could imagine having F1 scores for every class in a one-versus-all (OVA) fashion.  

Read more [[1]](https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest) 
[[2]](https://towardsdatascience.com/multi-class-metrics-made-simple-part-ii-the-F1-score-ebe8b2c2ca1).

""")

    write_instructor_expectation("""
This enables to set limits on the usage of F1 score and ensure the student understood it was particularly
suited in a binary setting, but needs to be thought again for different settings. 
More ideas [here](https://stats.stackexchange.com/a/121939).
""")

    st.markdown("""
$2.4)$ If I were to train a linear regression, would F1 score suit as a loss?
""")

    write_answer("""
As such, it's not so straightforward. Indeed, F1 score is not differentiable so computing gradients
will not be possible with such a loss. You would need 
to come up with a differentiable function that's close enough to F1 score. 

Nevertheless, it's ok 
to train your model with a loss but evaluate with different metrics (here, F1 score). In a business setting, it will
tend to always be the case (unless your stakeholders can map binary cross-entropy into a somewhat business performance 
indicator, which I doubt).

For your information, Eben et al. (2017) from Google suggested a solution to approximate F1 score with a differentiable
function in their 
[paper:](http://proceedings.mlr.press/v54/eban17a/eban17a.pdf) Scalable Learning of Non-Decomposable Objectives, 
*Proceedings of Machine Learning Research, 2017*. 
""")

    write_instructor_expectation("""
This is an advanced question, with pointers to research papers, for those that really want to deepen the question.
""")

    st.markdown("""
**Learn more:** another interesting performance measure for binary classifiers is area under the curve (AUC) of the 
receiver operating characteristic (ROC).
""")

    if st.button("🎈 That's all for this task. Click to celebrate!"):
        st.balloons()