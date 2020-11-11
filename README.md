[![GitHub size](https://github-size-badge.herokuapp.com/arnaudmiribel/exts_application.svg)](https://github.com/arnaudmiribel/exts_application/archive/master.zip)
[![HitCount](http://hits.dwyl.com/arnaudmiribel/exts_application.svg)](http://hits.dwyl.com/arnaudmiribel/exts_application)


# EXTS Application

This repository contains the source code of my application to [EPFL Extension School](https://www.extensionschool.ch/) job ad as an instructor for the Applied Data Science & Machine Learning course.

See it live here: [https://exts-application.herokuapp.com](https://exts-application.herokuapp.com)

# Getting started

## Dependencies

| Package                                   | Tested version |
|-------------------------------------------|----------------|
| streamlit                                 | 0.51.0         |
| numpy                                     | 1.17.4         |
| pandas                                    | 0.25.3         |
| plotly                                    | 4.4.1          |
| scitkit_learn                             | 0.22.1         |

## Installation

1. Clone the current repository
2. Install the requirements

```
pip install -r requirements.txt
```

3. Now open a terminal and run:

```
streamlit run exts_app.py
```

You should see the following output:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.86.27:8501
```

Now visit [http://localhost:8501](http://localhost:8501) and it should look like [that](exts-application.herokuapp.com)!